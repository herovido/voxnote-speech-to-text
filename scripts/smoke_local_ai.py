from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.local_analysis import RuleBasedMeetingAnalyzer
from backend.transcription import LocalWhisperTranscriber


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test faster-whisper with a local audio file.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--language", default="vi")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio.is_file():
        raise SystemExit(f"Audio file does not exist: {args.audio}")

    transcriber = LocalWhisperTranscriber(
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language or None,
        download_root=Path("runtime/models"),
        local_files_only=args.local_files_only,
        analyzer=RuleBasedMeetingAnalyzer(),
    )
    result = transcriber.transcribe(
        args.audio,
        lambda progress, stage: print(f"{progress}% {stage}", flush=True),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
