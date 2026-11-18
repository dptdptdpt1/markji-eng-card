"""在现有本地 Kokoro 环境中运行。"""
import hashlib
import html
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import wave
import zipfile

root = Path(__file__).resolve().parent.parent
library = root / "内容"
runtime = root / "local-tts"
output = root / "导出文件" / "Anki牌组"
cache = root / "音频"
output.mkdir(parents=True, exist_ok=True)
cache.mkdir(exist_ok=True)
os.environ.update(HF_HOME=str(runtime / "hf-cache"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
import espeakng_loader
dictionary = str(runtime / "espeak-ng-data")
assert (Path(dictionary) / "phontab").is_file() and len(dictionary.encode()) < 160
espeakng_loader.get_data_path = lambda: dictionary
os.environ.update(ESPEAK_DATA_PATH=dictionary, PHONEMIZER_ESPEAK_DATA_PATH=dictionary,
                  PHONEMIZER_ESPEAK_LIBRARY=espeakng_loader.get_library_path())
import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model

model = load_model(runtime / "models/Kokoro-82M-bf16")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def audio_for(english):
    filename = cache / (digest("af_heart/1.0/" + english)[:24] + ".wav")
    if not filename.exists():
        samples = []
        for result in model.generate(text=english, voice="af_heart", speed=1.0, lang_code="a"):
            mx.eval(result.audio)
            assert result.sample_rate == 24000
            samples.append(np.asarray(result.audio).reshape(-1))
        signal = np.concatenate(samples)
        assert np.isfinite(signal).all() and np.max(np.abs(signal)) > 0.02
        with wave.open(str(filename), "wb") as stream:
            stream.setparams((1, 2, 24000, 0, "NONE", ""))
            stream.writeframes((np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())
    with wave.open(str(filename)) as stream:
        assert (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) == (1, 2, 24000)
        assert 6000 < stream.getnframes() < 480000
    return filename


def build(level):
    title = "日常英语口语｜" + level
    pairs = []
    scenes = []
    source = library / (level + ".txt")
    scene = None
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("# "):
            scene = line[2:].strip()
            continue
        fields = [field.strip() for field in line.split("｜")]
        assert scene and len(fields) in (2, 3) and all(fields), (source, line)
        pairs.append(fields)
        scenes.append(scene)
    assert pairs and all(len({row[index] for row in pairs}) == len(pairs) for index in (0, 1))
    assets = [audio_for(row[1]) for row in pairs]
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        database = Path(tmp) / "collection.anki2"
        database.write_bytes((root / "工具" / "模板" / "sample.anki2").read_bytes())
        db = sqlite3.connect(database)
        for table in ("cards", "notes", "graves", "revlog"):
            db.execute("DELETE FROM " + table)
        conf, models, decks = map(json.loads, db.execute("SELECT conf,models,decks FROM col").fetchone())
        base = int(digest(title)[:9], 16) * 10000
        deck_id, model_id = base, base + 1
        now = int(time.time())
        deck = next(value for key, value in decks.items() if key != "1")
        desc = json.loads((library / "牌组简介.json").read_text())
        deck.update(id=deck_id, name=title, desc=desc[level], mod=now)
        note_model = next(iter(models.values()))
        note_model.update(id=model_id, name=title, did=deck_id, mod=now)
        note_model["tmpls"][0].update(qfmt="{{Front}}", afmt='{{FrontSide}}<hr id="answer">{{Back}}', did=None)
        conf.update(curDeck=deck_id, activeDecks=[deck_id], curModel=str(model_id), nextPos=len(pairs)+1)
        db.execute("UPDATE col SET mod=?,scm=?,conf=?,models=?,decks=?,tags='{}'",
                   (now*1000, now*1000, json.dumps(conf), json.dumps({model_id:note_model}),
                    json.dumps({1:decks["1"], deck_id:deck})))
        expected = []
        for index, (row, asset) in enumerate(zip(pairs, assets), 1):
            front, english, *note = row
            back = html.escape(english, quote=False) + " [sound:" + asset.name + "]"
            if note:
                back += "<br><br>" + html.escape(note[0], quote=False)
            fields = html.escape(front, quote=False) + "\x1f" + back
            expected.append(fields)
            checksum = int(hashlib.sha1(front.encode()).hexdigest()[:8], 16)
            note_id = int(digest(title + "/" + english)[:12], 16)
            db.execute("INSERT INTO notes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (note_id,digest(title+"/"+english)[:16],model_id,now,-1," "+scenes[index-1]+" ",fields,front,checksum,0,""))
            db.execute("INSERT INTO cards VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (note_id,note_id,deck_id,0,now,-1,0,0,index,0,0,0,0,0,0,0,0,""))
        db.commit()
        db.close()
        package = Path(tmp) / (title + ".apkg")
        manifest = {str(num):asset.name for num, asset in enumerate(assets)}
        with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(database, database.name)
            archive.writestr("media", json.dumps(manifest))
            for key, asset in zip(manifest, assets):
                archive.write(asset, key)
        with zipfile.ZipFile(package) as archive:
            assert archive.testzip() is None and json.loads(archive.read("media")) == manifest
            database.write_bytes(archive.read(database.name))
            db = sqlite3.connect(database)
            assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            assert db.execute("SELECT DISTINCT did FROM cards").fetchall() == [(deck_id,)]
            assert [row[0].strip() for row in db.execute("SELECT tags FROM notes JOIN cards ON notes.id=cards.nid ORDER BY due")] == scenes
            assert [row[0] for row in db.execute("SELECT flds FROM notes JOIN cards ON notes.id=cards.nid ORDER BY due")] == expected
            db.close()
            for key, asset in zip(manifest, assets):
                assert archive.read(key) == asset.read_bytes()
        package.replace(output / package.name)
    print(title, len(pairs), flush=True)


if __name__ == "__main__":
    for level in ("入门", "进阶"):
        build(level)
