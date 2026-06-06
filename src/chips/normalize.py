"""Romanian text normalization used by the CHiPS experiments.

The goal is conservative normalization: reduce technical variation while
preserving punctuation, whitespace, diacritics, and other character-level cues
that the models intentionally use.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

CEDILLA_TO_COMMA = str.maketrans({"Ş": "Ș", "ş": "ș", "Ţ": "Ț", "ţ": "ț"})


def normalize_text(text: str) -> str:
    """Normalize a Romanian text without removing stylistic evidence."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(CEDILLA_TO_COMMA)
    text = re.sub(r'[„“”]', '"', text)
    text = re.sub(r"[’‘]", "'", text)
    text = re.sub(r"[—–―]", "-", text)
    text = re.sub(r"\.(\s*\.){2,}", "…", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n+", "\n", text)
    return text


def normalize_file(input_path: Path, output_path: Path) -> None:
    text = input_path.read_text(encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(normalize_text(text), encoding="utf-8")


def normalize_directory(input_dir: Path, output_dir: Path, suffix: str = ".txt") -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(input_dir.rglob(f"*{suffix}")):
        if not path.is_file():
            continue
        rel = path.relative_to(input_dir)
        normalize_file(path, output_dir / rel)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Romanian text files for CHiPS.")
    parser.add_argument("input", type=Path, help="Input .txt file or directory.")
    parser.add_argument("output", type=Path, help="Output .txt file or directory.")
    args = parser.parse_args()
    if args.input.is_dir():
        count = normalize_directory(args.input, args.output)
        print(f"Normalized {count} text files into {args.output}")
    else:
        normalize_file(args.input, args.output)
        print(f"Normalized {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
