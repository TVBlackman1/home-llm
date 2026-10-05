from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ollama_runner.factory import close_resources, default_pipeline, semantic_pipeline
from ollama_runner.ha.client import HaClient
from ollama_runner.ha.execute import failure_reason
from ollama_runner.ha.route import ExecutionRouter
from ollama_runner.ha.normalize import load_inventory
from ollama_runner.request import configure_request_logging
from ollama_runner.resolve.capability import CapabilityResolver
from ollama_runner.semantic_pipeline import SemanticPipeline
from ollama_runner.settings import get_settings
from ollama_runner.sinks.jsonprint import PrintSink
from ollama_runner.types import Command, Result


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
        help="json keeps the benchmark pipeline; ha runs the semantic pipeline against Home Assistant",
    )
    parser.add_argument(
        "--model",
        help="Override OLLAMA_MODEL from .env",
    )

    args = parser.parse_args()
    if args.sink == "ha":
        if args.audio is not None:
            raise SystemExit("Home Assistant mode takes text")
        _run_home(args, settings)
        return

    pipeline = default_pipeline(model=args.model)
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


def run_home_request(client, text: str, *, model: str | None = None, nlu=None) -> Result:
    """One utterance against one fresh Home Assistant inventory.

    The client and the language model stay process-scoped. A failed refresh
    does not reuse an older device map.
    """

    try:
        inventory = load_inventory(client)
    except Exception as exc:
        return Result(
            ok=False,
            command=Command(device_id="", action=""),
            error="unavailable",
            payload={"status": "unavailable", "reason": failure_reason(exc)},
        )
    executor = ExecutionRouter(inventory.registry, inventory.bindings, client)
    members = {
        entity_id: binding.members
        for entity_id, binding in inventory.bindings.items()
        if binding.members
    }
    if nlu is None:
        pipeline = semantic_pipeline(
            model=model,
            registry=inventory.registry,
            executor=executor,
            members=members,
        )
    else:
        pipeline = SemanticPipeline(
            nlu,
            CapabilityResolver(inventory.registry, members=members),
            executor,
            inventory.registry,
        )
    return pipeline.run(text, _QuietSink(), state=inventory.state)


def _run_home(args, settings) -> None:
    client = HaClient(settings)
    try:
        configure_request_logging()

        def once(text: str) -> bool:
            return run_home_request(client, text, model=args.model).ok

        if args.text is not None:
            raise SystemExit(0 if once(args.text) else 1)
        if args.file is not None:
            raise SystemExit(0 if once(args.file.read_text(encoding="utf-8")) else 1)
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text:
                once(text)
    finally:
        client.close()
        close_resources()


class _QuietSink:
    """Accept a resolved command without printing the legacy payload."""

    def send(self, command: Command) -> Result:
        return Result(ok=True, command=command)


if __name__ == "__main__":
    main()
