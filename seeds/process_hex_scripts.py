#!/usr/bin/env python3

"""Process ARC seed scripts and concatenate their contents.

For each Python file in this directory whose name is an 8-digit hex puzzle ID:

1. Remove all import statements.
2. Drop all code starting from the first ``def generate_input`` definition to the end.
3. Rename ``def main`` to ``def transform_<puzzle_id>``.

After transforming the files in place, concatenate their contents into ``all.py``.
"""

from __future__ import annotations

import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
HEX_PATTERN = re.compile(r"^[0-9a-f]{8}\.py$")
TRAINING_DIR = ROOT.parent.parent / "ARC-AGI" / "data" / "training"


try:
    from test import show_puzzle
except ImportError as exc:  # pragma: no cover - fail fast if environment changes
    raise RuntimeError("Unable to import show_puzzle from test.py") from exc


class ProcessingError(RuntimeError):
    """Raised when a file cannot be processed as requested."""


def iter_hex_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("*.py")):
        if HEX_PATTERN.match(path.name):
            yield path


def strip_imports_and_truncate(content: str) -> str:
    lines = content.splitlines()
    output_lines: list[str] = []

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("def generate_input"):
            break
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        output_lines.append(line)

    return "\n".join(output_lines).rstrip() + "\n"


def rename_main(content: str, puzzle_id: str) -> str:
    pattern = re.compile(r"(?m)^(\s*)def\s+main(\s*\()");
    replacement = rf"\1def transform_{puzzle_id}\2"
    content, count = pattern.subn(replacement, content)
    if count == 0:
        raise ProcessingError(f"No main function found to rename in {puzzle_id}.py")
    return content.lstrip("\n")


def render_puzzle_display(puzzle_id: str) -> str:
    json_path = TRAINING_DIR / f"{puzzle_id}.json"
    if not json_path.exists():
        raise ProcessingError(f"Puzzle data not found: {json_path}")

    with json_path.open() as f:
        puzzle = json.load(f)

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        show_puzzle(puzzle, ID=puzzle_id, show_sections=["train"], max_pairs=3)

    display = buffer.getvalue().rstrip()
    return display + "\n" if display else ""


def process_file(path: Path) -> str:
    original = path.read_text()
    truncated = strip_imports_and_truncate(original)
    renamed = rename_main(truncated, path.stem)
    display = render_puzzle_display(path.stem)
    display = display.rstrip()
    renamed = renamed.rstrip()

    if display:
        comment_block = ('""" ' + display + ' """').strip("\n")
        combined = f"{comment_block}\n\n{renamed}"
    else:
        combined = renamed

    return combined.rstrip() + "\n"


def concatenate(contents: Iterable[str], destination: Path) -> None:
    joined = "\n\n".join(text.rstrip() for text in contents) + "\n"
    destination.write_text(joined)


def main() -> None:
    processed_contents: list[str] = []
    errors: list[str] = []

    for file_path in iter_hex_files(ROOT):
        try:
            processed_contents.append(process_file(file_path))
        except ProcessingError as exc:
            errors.append(str(exc))

    if processed_contents:
        concatenate(processed_contents, ROOT / "all.py")

    if errors:
        for message in errors:
            print(f"Warning: {message}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Processed {len(processed_contents)} files and wrote all.py")


if __name__ == "__main__":
    main()


