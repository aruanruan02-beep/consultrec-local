# 本地咨询录音处理系统

一个运行在本地 Mac 上的离线网页工具，用于把心理咨询录音转换为结构化临床记录。

## 能做什么

- 浏览器上传单个音频文件：`mp3` / `wav` / `m4a`
- 按 Case ID 和 Session 日期自动归档
- 调用本地 WhisperX 流程，生成带时间轴和说话人标签的逐字稿
- 支持把“说话人 1 / 说话人 2”手动映射为咨询师 / 来访者
- 在生成临床记录前校对说话人角色
- 动态加载独立 Prompt 文件，调用本地 LLM 生成：
  - SOAP 记录
  - Session Summary
- 输出每个 session 的 Markdown 与 JSON

> 处理过程在本机运行。WhisperX 首次运行会下载 ASR / diarization 模型；说话人分离需要配置 Hugging Face read token，并接受 pyannote diarization 模型的使用条款。

## 快速开始

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

启动本地网页界面：

```bash
python3 -m consultrec serve
```

然后打开：

```text
http://127.0.0.1:8000
```

查看命令行帮助：

```bash
python3 -m consultrec --help
```

示例：

```bash
python3 -m consultrec process ./audio/session01.m4a \
  --session-id session01 \
  --output-dir ./sessions \
  --asr-command '.venv/bin/python scripts/transcribe_whisperx.py --audio {audio} --output {asr_output_json} --model small --language zh --device cpu --compute-type int8 --min-speakers 2 --max-speakers 2' \
  --llm-command 'llama-cli -m ./models/local-model.gguf -f {prompt_file}'
```

如果本地转写结果已经是 JSON，可以跳过 ASR：

```bash
python3 -m consultrec process ./audio/session01.m4a \
  --session-id session01 \
  --output-dir ./sessions \
  --transcript-json ./examples/transcript.example.json \
  --llm-command 'llama-cli -m ./models/local-model.gguf -f {prompt_file}'
```

## Prompt

Prompt 独立存放在：

- `prompts/clinical_note_prompt.md`

运行时会动态读取。以后修改临床记录格式或约束时，只需要改 Prompt 文件，不需要改代码。

## 输出

每个 session 会生成：

- `data/cases/{case_id}/case.json`
- `data/cases/{case_id}/sessions/{session_date}_{session_id}/audio.*`
- `data/cases/{case_id}/sessions/{session_date}_{session_id}/transcript.json`
- `data/cases/{case_id}/sessions/{session_date}_{session_id}/clinical_note.json`
- `data/cases/{case_id}/sessions/{session_date}_{session_id}/session.md`
- `data/cases/{case_id}/sessions/{session_date}_{session_id}/session.json`

## 说话人区分

默认转写命令使用 WhisperX 一次生成文字、时间戳和 speaker 字段：

```json
{
  "speaker": "说话人 1",
  "text": "..."
}
```

前端结果页会保留 `说话人 1 / 说话人 2`，由用户手动映射为 `咨询师 / 来访者`，并支持逐段继续修改。
