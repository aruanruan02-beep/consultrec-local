# 本地咨询录音处理系统

一个运行在本地 Mac 上的离线网页工具，用于把心理咨询录音转换为结构化临床记录。

## 能做什么

- 浏览器上传单个音频文件：`mp3` / `wav` / `m4a`
- 按 Case ID 和 Session 日期自动归档
- 调用本地离线语音转写工具，生成带时间轴逐字稿
- 支持咨询师 / 来访者角色标注
- 在生成临床记录前校对说话人角色
- 动态加载独立 Prompt 文件，调用本地 LLM 生成：
  - SOAP 记录
  - Session Summary
- 输出每个 session 的 Markdown 与 JSON

> 本项目不包含云端 API 调用，也不需要网络。你需要在本机提前准备可用的离线 ASR 与本地 LLM 命令，例如 whisper.cpp 与 llama.cpp。

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
  --asr-command 'whisper-cli -m ./models/ggml-large-v3-turbo.bin -f {audio} -oj -of {asr_output_stem}' \
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

## 角色区分

工具支持三种来源：

- 转写 JSON 中已经包含 `speaker` 字段
- 提供离线说话人区分命令：`--diarization-command`
- 使用 `--roles-mode alternating` 做本地草稿标注

建议正式使用时接入本地 diarization 工具，并用 `--therapist-speaker` / `--client-speaker` 固定说话人映射。
