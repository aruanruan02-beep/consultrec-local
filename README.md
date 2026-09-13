# 本地咨询录音处理系统（AI 部署说明）

本仓库是一个仅在本机运行的 Mac 网页工具：它可将单个咨询录音转为带时间轴的逐字稿，支持说话人角色校对，并在本机生成 SOAP 记录与 Session Summary。

本 README 面向协助使用者部署项目的 AI 代理（如 Codex），而不是面向终端使用者的操作手册。面向人的部署说明由项目维护者单独提供。

## 功能范围

- 上传单个 `mp3`、`wav` 或 `m4a` 录音；
- 按 Case ID 与会谈日期归档；
- 使用 MLX Whisper 在 Apple 芯片 Mac 本机生成逐字稿；
- 可选使用 pyannote 自动区分两名说话人；
- 允许使用者手动将说话人映射为“咨询师 / 来访者”；
- 通过本机 Ollama 模型生成 SOAP 记录与 Session Summary；
- 输出每次会谈的 Markdown 与 JSON。

## 非协商的隐私与安全边界

1. 录音、逐字稿、临床记录及 `data/` 中的任何内容必须留在本机。不得上传、发送、复制到聊天、提交至 Git 或暴露到局域网/公网。
2. 服务必须仅绑定 `127.0.0.1`；不得改为 `0.0.0.0`。
3. 不得读取、显示、索取或提交用户的 Hugging Face token。需要 token 时，要求用户自行在本机安全保存；AI 只确认配置是否可用。
4. 不得提交 `.env`、`config/local_settings.json`、`data/`、`测试数据/`、`.venv/`、`.pyannote-venv/`、`node_modules/` 或任何音视频文件。
5. 不得把模型输出当作诊断、治疗建议或专业判断。

## 目标环境与前置检查

- 目标平台：Apple 芯片 Mac（M1 / M2 / M3 / M4 等）；默认 `mlx-whisper` 不适用于 Windows 或 Intel Mac。
- 在开始前检查至少 20 GB 可用磁盘空间；若启用自动说话人区分并保存多次会谈资料，建议至少 30 GB。
- 检查 Python 3.10+、Ollama 与网络连接。首次运行需下载 Python 依赖和本地模型。
- 若缺少 Ollama、Python 或系统工具，说明将要安装的内容与用途，并在执行会修改系统的软件安装前征得用户确认。

## AI 部署流程

从仓库根目录执行，按此顺序完成：

1. 检查项目文件、macOS 架构、可用磁盘空间和 Python 版本。
2. 创建 `.venv`，运行 `bash scripts/install_local_models.sh` 安装 Python 依赖。
3. 确保 Ollama 已安装和运行，下载默认本地模型：`qwen2.5:7b-instruct`。
4. 创建本机配置 `config/local_settings.json`（该文件仅保留在本机），至少包含：

   ```json
   {
     "whisper_command": ".venv/bin/python scripts/transcribe_mlx_whisper.py --audio {audio} --output {transcript_json} --model mlx-community/whisper-small-mlx --language zh",
     "diarization_command": "",
     "llm_command": ".venv/bin/python scripts/generate_note_ollama.py --prompt-file {prompt_file} --model qwen2.5:7b-instruct"
   }
   ```

5. 启动服务：`.venv/bin/python -m consultrec serve`。
6. 验证 `http://127.0.0.1:8000` 可以打开；告知用户本地访问地址以及如何停止服务。

不要尝试处理用户录音来作为部署验证；页面可访问且配置/命令有效即完成基础验证。

## 可选：自动说话人区分

仅在用户明确需要时启用。

1. 提醒用户先自行登录 [Hugging Face 的 pyannote 模型页面](https://huggingface.co/pyannote/speaker-diarization-community-1)，接受模型条款并创建自己的 read token。
2. 用户自行在本机保存 token 后，安装隔离的 pyannote 环境：`bash scripts/install_pyannote_env.sh`。
3. 在本机配置中设置：

   ```json
   {
     "diarization_command": ".pyannote-venv/bin/python scripts/diarize_pyannote.py --audio {audio} --output {diarization_json} --num-speakers 2"
   }
   ```

4. 重启本地服务并验证配置错误信息不会泄露 token。

若用户没有完成 Hugging Face 授权或不想配置 token，保留 `diarization_command` 为空；系统仍可使用手动说话人角色映射。

## 项目入口

- Web 服务：`consultrec/web.py`
- 命令行入口：`python -m consultrec serve`
- 默认本机配置：`consultrec/config.py`
- MLX Whisper 转写脚本：`scripts/transcribe_mlx_whisper.py`
- pyannote 说话人区分脚本：`scripts/diarize_pyannote.py`
- Ollama 记录生成脚本：`scripts/generate_note_ollama.py`
- 临床记录 Prompt：`prompts/clinical_note_prompt.md`

## 完成标准

部署完成时，向用户简洁报告：已安装的组件、服务本地地址、是否启用了自动说话人区分，以及任何需要用户本人完成的授权步骤。不要报告、打印或索取任何敏感数据。
