import argparse
from pathlib import Path

from .pipeline import process_session


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="consultrec",
        description="本地咨询录音处理系统：音频转写、角色标注、SOAP 与 Session Summary 输出。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="启动本地网页界面")
    serve.add_argument("--host", default="127.0.0.1", help="监听地址，默认只允许本机访问")
    serve.add_argument("--port", type=int, default=8000, help="端口")

    process = subparsers.add_parser("process", help="处理单个音频文件")
    process.add_argument("audio", type=Path, help="音频文件路径")
    process.add_argument("--session-id", required=True, help="session 输出文件名标识")
    process.add_argument("--output-dir", type=Path, default=Path("sessions"), help="输出目录")
    process.add_argument(
        "--prompt",
        type=Path,
        default=Path("prompts/clinical_note_prompt.md"),
        help="临床记录 Prompt 文件",
    )
    process.add_argument("--asr-command", help="离线转写命令模板")
    process.add_argument("--transcript-json", type=Path, help="已有逐字稿 JSON")
    process.add_argument("--llm-command", required=True, help="本地 LLM 命令模板，支持 {prompt_file}")
    process.add_argument(
        "--roles-mode",
        choices=["existing", "alternating"],
        default="existing",
        help="角色标注方式。existing 使用已有 speaker 字段；alternating 用于草稿。",
    )
    process.add_argument("--diarization-command", help="离线说话人区分命令模板")
    process.add_argument("--therapist-speaker", help="把该 speaker 映射为 Therapist")
    process.add_argument("--client-speaker", help="把该 speaker 映射为 Client")

    args = parser.parse_args()
    if args.command == "serve":
        try:
            import uvicorn
        except ModuleNotFoundError as exc:
            raise SystemExit("请先安装依赖：python3 -m pip install -r requirements.txt") from exc
        uvicorn.run("consultrec.web:app", host=args.host, port=args.port, reload=False)
    elif args.command == "process":
        process_session(
            audio_path=args.audio,
            output_dir=args.output_dir,
            session_id=args.session_id,
            prompt_path=args.prompt,
            llm_command=args.llm_command,
            asr_command=args.asr_command,
            transcript_json=args.transcript_json,
            roles_mode=args.roles_mode,
            diarization_command=args.diarization_command,
            therapist_speaker=args.therapist_speaker,
            client_speaker=args.client_speaker,
        )
