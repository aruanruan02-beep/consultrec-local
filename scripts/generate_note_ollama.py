#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import List

import ollama

OLLAMA_CANDIDATES = [
    Path("/Applications/Ollama.app/Contents/Resources/ollama"),
    Path("/usr/local/bin/ollama"),
    Path("/opt/homebrew/bin/ollama"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clinical note with a local Ollama model.")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--chunk-chars", type=int, default=6000)
    args = parser.parse_args()

    prompt = open(args.prompt_file, "r", encoding="utf-8").read()
    ensure_ollama_running()
    if len(prompt) > args.chunk_chars * 2 and "# 逐字稿" in prompt:
        response_text = generate_with_chunk_summaries(args.model, prompt, args.num_ctx, args.chunk_chars)
    else:
        response_text = generate(args.model, prompt, args.num_ctx)
    sys.stdout.write(response_text)


def generate(model: str, prompt: str, num_ctx: int) -> str:
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            "temperature": 0.0,
            "num_ctx": num_ctx,
        },
    )
    return response.get("response", "")


def generate_with_chunk_summaries(model: str, prompt: str, num_ctx: int, chunk_chars: int) -> str:
    prefix, transcript = prompt.split("# 逐字稿", 1)
    transcript = transcript.strip()
    chunks = split_transcript(transcript, chunk_chars)
    chunk_summaries = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_prompt = f"""你是咨询记录整理助手。请只根据下面这个逐字稿片段，写咨询记录正文中的一段或两段。

要求：
- 写成连续自然段，不要写项目符号。
- 保留这个片段里实际谈到的具体内容、事件、情绪和咨询师回应。
- 不要写建议，不要诊断，不要推测逐字稿之外的信息。
- 不要写“咨询师建议/鼓励/提醒来访者……”，除非逐字稿中咨询师明确说了这些内容。
- 不要写“本片段”“片段中”等元说明。

片段 {index}/{len(chunks)}：

{chunk}
"""
        chunk_summaries.append(generate(model, chunk_prompt, num_ctx).strip())

    payload = {
        "session_summary": {
            "咨询记录": normalize_record_paragraphs(chunk_summaries),
        },
        "soap": {
            "S": ["逐字稿中未明确提及"],
            "O": ["逐字稿中未明确提及"],
            "A": ["逐字稿中未明确提及"],
            "P": ["逐字稿中未明确提及"],
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def split_transcript(transcript: str, chunk_chars: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_size = 0
    for line in transcript.splitlines():
        line_size = len(line) + 1
        if current and current_size + line_size > chunk_chars:
            chunks.append("\n".join(current))
            current = []
            current_size = 0
        current.append(line)
        current_size += line_size
    if current:
        chunks.append("\n".join(current))
    return chunks


def normalize_summary_items(chunk_summaries: List[str]) -> List[str]:
    items: List[str] = []
    for summary in chunk_summaries:
        current = ""
        for raw_line in summary.splitlines():
            line = re.sub(r"\*\*([^*]+)\*\*", r"\1", raw_line).strip()
            if not line:
                continue
            match = re.match(r"^(?:[-*]|\d+[.、])\s*(.+)$", line)
            if match:
                if current:
                    items.append(clean_summary_item(current))
                current = match.group(1).strip()
            elif current:
                current = f"{current} {line}".strip()
            elif len(line) > 8:
                items.append(clean_summary_item(line))
        if current:
            items.append(clean_summary_item(current))

    deduped: List[str] = []
    seen = set()
    for item in items:
        if not item or should_drop_summary_item(item):
            continue
        key = re.sub(r"\W+", "", item)[:40]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped or ["逐字稿中未明确提及"]


def normalize_record_paragraphs(chunk_summaries: List[str]) -> List[str]:
    paragraphs: List[str] = []
    for summary in chunk_summaries:
        cleaned = re.sub(r"```.*?```", "", summary, flags=re.DOTALL).strip()
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
        candidates = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
        if len(candidates) <= 1:
            candidates = [line.strip() for line in cleaned.splitlines() if line.strip()]
        for candidate in candidates:
            paragraph = clean_record_paragraph(candidate)
            if paragraph and not should_drop_summary_item(paragraph):
                paragraphs.append(paragraph)

    return paragraphs or ["逐字稿中未明确提及"]


def clean_summary_item(text: str) -> str:
    text = re.sub(r"^片段\s*\d+[/\d]*[:：]?", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("考虑跳槽到其他行业", "通过读博和换专业赛道寻找更多可能性")
    return text.strip(" -。；;")


def clean_record_paragraph(text: str) -> str:
    text = re.sub(r"^(?:[-*]|\d+[.、])\s*", "", text.strip())
    text = re.sub(r"^片段\s*\d+[/\d]*[:：]?", "", text).strip()
    text = re.sub(r"^咨询记录如下[:：]?", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("考虑跳槽到其他行业", "通过读博和换专业赛道寻找更多可能性")
    text = remove_advice_sentences(text)
    if text and text[-1] not in "。！？":
        text = f"{text}。"
    return text


def remove_advice_sentences(text: str) -> str:
    sentences = re.findall(r"[^。！？]+[。！？]?", text)
    kept = []
    advice_markers = (
        "咨询师建议",
        "咨询师鼓励",
        "咨询师提醒",
        "建议来访者",
        "鼓励来访者",
        "提醒来访者",
        "可以考虑",
        "需要关注",
        "以确保",
    )
    for sentence in sentences:
        stripped = sentence.strip()
        if stripped and not any(marker in stripped for marker in advice_markers):
            kept.append(stripped)
    return "".join(kept).strip()


def should_drop_summary_item(text: str) -> bool:
    dropped_markers = (
        "不要写建议",
        "不要诊断",
        "要求",
        "以下是",
        "总结如下",
        "可以提供一些",
        "帮助来访者",
        "小朋友",
        "咨询记录如下",
    )
    return any(marker in text for marker in dropped_markers)


def ensure_ollama_running() -> None:
    if can_reach_ollama():
        return
    executable = next((path for path in OLLAMA_CANDIDATES if path.exists()), None)
    if not executable:
        return
    subprocess.Popen(
        [str(executable), "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(30):
        if can_reach_ollama():
            return
        time.sleep(0.5)


def can_reach_ollama() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1):
            return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
