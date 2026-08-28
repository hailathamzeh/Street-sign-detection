#!/usr/bin/env python3
"""Static validation for the repository and its notebooks."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 25 * 1024 * 1024

REQUIRED_FILES = (
    "README.md",
    ".gitignore",
    ".env.example",
    "requirements.txt",
    "config/jordanian_traffic_signs.yaml",
    "data/README.md",
    "models/README.md",
    "assets/ATTRIBUTION.md",
    "outputs/yolov5/results.csv",
    "notebooks/01_exploring_data.ipynb",
    "notebooks/02_custom_cnn_lime.ipynb",
    "notebooks/03_vgg16_lime.ipynb",
    "notebooks/04_yolov5_shap.ipynb",
)

FORBIDDEN_NAMES = {
    "kaggle.json",
    ".env",
    ".DS_Store",
    "Thumbs.db",
}

FORBIDDEN_SUFFIXES = {
    ".rar",
    ".zip",
    ".7z",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".keras",
    ".onnx",
    ".db",
    ".sqlite",
    ".sqlite3",
}

TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".csv",
    ".toml",
    ".ini",
    ".cfg",
    ".example",
    ".gitignore",
}

SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?:api[_-]?key|client[_-]?secret|access[_-]?token|password|kaggle_key)"
    r"\s*[:=]\s*[\"']([^\"'\s]{8,})[\"']"
)

LOCAL_PATH_PATTERNS = (
    re.compile("/" + "content" + "/"),
    re.compile(r"/kaggle/(?:input|working)/"),
    re.compile(r"[A-Za-z]:[\\/]Users[\\/]"),
    re.compile(r"/(?:home|Users)/[^/\s]+/"),
)


def repository_files() -> list[Path]:
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
    return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def text_from_notebook(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in notebook.get("cells", []):
        chunks.append("".join(cell.get("source", [])))
        for output in cell.get("outputs", []):
            text = output.get("text")
            if text:
                chunks.append("".join(text) if isinstance(text, list) else str(text))
            for mime, value in output.get("data", {}).items():
                if mime.startswith("text/"):
                    chunks.append("".join(value) if isinstance(value, list) else str(value))
    return "\n".join(chunks)


def validate_notebook(path: Path, errors: list[str]) -> None:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"Invalid notebook JSON: {path.relative_to(ROOT)} ({exc})")
        return

    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        errors.append(f"Unsupported notebook structure: {path.relative_to(ROOT)}")
        return

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        cleaned = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith(("!", "%"))
        )
        if not cleaned.strip():
            continue
        try:
            ast.parse(cleaned)
        except SyntaxError as exc:
            errors.append(
                f"Python syntax error in {path.relative_to(ROOT)}, cell {index}: {exc.msg}"
            )


def scan_text(path: Path, text: str, errors: list[str]) -> None:
    relative = path.relative_to(ROOT)
    if "\u2014" in text:
        errors.append(f"Em dash found in {relative}")
    allowed_placeholders = {
        "your_key",
        "your_api_key",
        "replace_me",
        "example_value",
    }
    for match in SECRET_ASSIGNMENT.finditer(text):
        value = match.group(1).lower()
        if value not in allowed_placeholders and not value.startswith(("<", "${")):
            errors.append(f"Possible embedded credential found in {relative}")
            break
    for pattern in LOCAL_PATH_PATTERNS:
        if pattern.search(text):
            errors.append(f"Hard-coded machine path found in {relative}: {pattern.pattern}")


def validate_readme_images(errors: list[str]) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for target in re.findall(r"!\[[^\]]*\]\(([^)\s]+)", readme):
        if re.match(r"https?://", target):
            continue
        if not (ROOT / target).is_file():
            errors.append(f"README image is missing: {target}")


def validate_dataset(errors: list[str]) -> None:
    root = ROOT / "data" / "Final_Dataset"
    required = [
        root / category / split
        for category in ("images", "labels")
        for split in ("train", "val", "test")
    ]
    for directory in required:
        if not directory.is_dir():
            errors.append(f"Dataset directory is missing: {directory}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-dataset",
        action="store_true",
        help="Also require the external Final_Dataset folder structure.",
    )
    args = parser.parse_args()

    errors: list[str] = []
    files = repository_files()

    for required in REQUIRED_FILES:
        if not (ROOT / required).is_file():
            errors.append(f"Required file is missing: {required}")

    for path in files:
        relative = path.relative_to(ROOT)
        if any(part in {".ipynb_checkpoints", "__pycache__", "runs"} for part in relative.parts):
            errors.append(f"Generated directory is present: {relative}")
        if path.name in FORBIDDEN_NAMES or path.name.startswith("events.out.tfevents"):
            errors.append(f"Forbidden file is present: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"Forbidden generated or binary file is present: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"File exceeds 25 MB validation limit: {relative}")

        if path.suffix.lower() == ".ipynb":
            validate_notebook(path, errors)
            scan_text(path, text_from_notebook(path), errors)
        elif path.suffix.lower() in TEXT_SUFFIXES or path.name in {"README.md", ".gitignore"}:
            try:
                scan_text(path, path.read_text(encoding="utf-8"), errors)
            except UnicodeDecodeError:
                errors.append(f"Expected UTF-8 text file: {relative}")

    validate_readme_images(errors)
    if args.check_dataset:
        validate_dataset(errors)

    if errors:
        print("Repository validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Repository validation passed for {len(files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
