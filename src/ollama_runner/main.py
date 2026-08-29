from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ollama_runner.factory import close_resources, default_pipeline
from ollama_runner.settings import get_settings
from ollama_runner.sinks.ha import HomeAssistant
from ollama_runner.sinks.jsonprint import PrintSink


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Raw command text")
    source.add_argument("--file", type=Path, help="UTF-8 text file")
    source.add_argument("--audio", type=Path, help="Audio file for Whisper")

    parser.add_argument(
        "--sink",
        choices=("json", "ha"),
        default="json",
        help="json prints Command, ha sends to the Home Assistant stub",
    )
    parser.add_argument(
        "--model",
        help="Override OLLAMA_MODEL from .env",
    )

    args = parser.parse_args()
    pipeline = default_pipeline(model=args.model)

    if args.sink == "ha":
        sink = HomeAssistant(
            host=settings.ha_host,
            port=settings.ha_port,
            token=settings.ha_token,
        )
    else:
        sink = PrintSink()

    try:
        if args.text is not None:
            result = pipeline.from_text(args.text).into(sink).run()
            raise SystemExit(0 if result.ok else 1)

        if args.file is not None:
            result = pipeline.from_file(args.file).into(sink).run()
            raise SystemExit(0 if result.ok else 1)

        if args.audio is not None:
            result = pipeline.from_audio(args.audio).into(sink).run()
            raise SystemExit(0 if result.ok else 1)

        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not text:
                continue

            result = pipeline.from_text(text).into(sink).run()
            if not result.ok:
                print(result.error, file=sys.stderr)
    finally:
        close_resources()


if __name__ == "__main__":
    main()
