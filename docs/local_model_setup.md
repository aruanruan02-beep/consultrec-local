# 本地模型安装说明

当前推荐的轻量本地方案：

- 转写：`faster-whisper`，本地 CPU 运行，兼容性更稳
- LLM：Ollama 本地运行 `qwen2.5:7b-instruct`

## 1. 安装 Python 依赖

在项目根目录运行：

```bash
bash scripts/install_local_models.sh
```

## 2. 安装 Ollama

如果本机还没有 Ollama，请安装 Mac 版：

```text
https://ollama.com/download/mac
```

安装并启动 Ollama 后，下载本地 LLM：

```bash
ollama pull qwen2.5:7b-instruct
```

## 3. 更新配置

把 `config/local_settings.json` 改成：

```json
{
  "data_root": "data",
  "whisper_command": ".venv/bin/python scripts/transcribe_faster_whisper.py --audio {audio} --output {transcript_json} --model medium",
  "diarization_command": "",
  "llm_command": ".venv/bin/python scripts/generate_note_ollama.py --prompt-file {prompt_file} --model qwen2.5:7b-instruct",
  "clinical_prompt_path": "prompts/clinical_note_prompt.md"
}
```

## 4. 启动网页工具

```bash
.venv/bin/python -m consultrec serve
```

打开：

```text
http://127.0.0.1:8000
```

## 说明

- 首次转写时，`faster-whisper` 会下载 Whisper 模型到本机缓存。
- 首次运行 LLM 前，需要先执行 `ollama pull qwen2.5:7b-instruct`。
- 说话人区分目前保留为校对流程；自动 diarization 可以后续接入本地命令。
