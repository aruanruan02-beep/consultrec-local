# 本地模型安装说明

当前推荐的本地方案：

- 转写和说话人区分：`WhisperX`，本地 CPU 运行，首次运行自动下载模型
- LLM：Ollama 本地运行 `qwen2.5:7b-instruct`

## 1. 安装 Python 依赖

在项目根目录运行：

```bash
bash scripts/install_local_models.sh
```

## 2. 配置 Hugging Face Token

WhisperX 的说话人分离需要 pyannote 模型。请先：

- 创建 Hugging Face read token
- 接受 `pyannote/speaker-diarization-community-1` 的模型使用条款
- 在启动服务的终端里设置环境变量：

```bash
export HF_TOKEN="你的 Hugging Face read token"
```

也可以使用 `HUGGINGFACE_TOKEN`。

## 3. 安装 Ollama

如果本机还没有 Ollama，请安装 Mac 版：

```text
https://ollama.com/download/mac
```

安装并启动 Ollama 后，下载本地 LLM：

```bash
ollama pull qwen2.5:7b-instruct
```

## 4. 更新配置

把 `config/local_settings.json` 改成：

```json
{
  "data_root": "data",
  "whisper_command": ".venv/bin/python scripts/transcribe_mlx_whisper.py --audio {audio} --output {transcript_json} --model mlx-community/whisper-small-mlx --language zh",
  "diarization_command": ".pyannote-venv/bin/python scripts/diarize_pyannote.py --audio {audio} --output {diarization_json} --num-speakers 2",
  "llm_command": ".venv/bin/python scripts/generate_note_ollama.py --prompt-file {prompt_file} --model qwen2.5:7b-instruct",
  "clinical_prompt_path": "prompts/clinical_note_prompt.md"
}
```

## 5. 启动网页工具

```bash
.venv/bin/python -m consultrec serve
```

打开：

```text
http://127.0.0.1:8000
```

## 说明

- 首次转写时，`WhisperX` 会下载 Whisper / alignment / diarization 模型到本机缓存。
- 首次运行 LLM 前，需要先执行 `ollama pull qwen2.5:7b-instruct`。
- 说话人标签会先显示为 `说话人 1 / 说话人 2`，请在前端手动映射为 `咨询师 / 来访者`。
