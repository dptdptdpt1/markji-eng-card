"""生成墨记文本；可传入平台导出文件以更新音频 ID。"""
import hashlib
import json
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
dest = root / "导出文件" / "墨记文本"
mapping_path = root / "工具" / "墨记音频映射.json"
mapping = json.loads(mapping_path.read_text())


def audio_name(english):
    # 历史缓存键，不代表实际音色；新音频的模型与参数见音频/生成记录.json。
    return hashlib.sha256(("af_heart/1.0/" + english).encode()).hexdigest()[:24] + ".wav"


for argument in sys.argv[1:]:
    path = Path(argument)
    sections = re.split(r"(?m)^### [^\n]+$", path.read_text())[1:]
    assert sections, f"文件缺少章节：{argument}"
    for block in (card for section in sections for card in re.split(r"(?m)^===$", section)):
        if not block.strip():
            continue
        front, answer = block.strip().split("\n---\n", 1)
        match, = re.finditer(r"\[Audio#(?:A,)?ID/([^#\]]+)#([^\]]*)\]", answer)
        english = match[2] or answer[:match.start()].strip()
        mapping[audio_name(english)] = match[1]

outputs = {}
used = {}
for level in ("入门", "进阶"):
    chapters = {}
    chapter = None
    for line in (root / "内容" / (level + ".txt")).read_text().splitlines():
        if line.startswith("# "):
            chapter = line[2:].strip()
            assert chapter and chapter not in chapters
            chapters[chapter] = []
            continue
        if not line.strip():
            continue
        front, english, *note = [field.strip() for field in line.split("｜")]
        assert chapter and front and english and len(note) <= 1
        name = audio_name(english)
        assert name in mapping, f"请先导入新音频，再用平台导出文件更新映射：{english}"
        audio = mapping[name]
        used[name] = audio
        chapters[chapter].append("\n".join([front, "---", f"[Audio#A,ID/{audio}#{english}]", *note]))
    result = [f"### {name}\n\n" + "\n===\n".join(rows) for name, rows in chapters.items()]
    outputs[level] = "\n\n".join(result) + "\n"
    print(level, {name: len(rows) for name, rows in chapters.items()})

dest.mkdir(parents=True, exist_ok=True)
for level, text in outputs.items():
    (dest / (level + ".txt")).write_text(text)
mapping_path.write_text(json.dumps(used, ensure_ascii=False, indent=2) + "\n")
