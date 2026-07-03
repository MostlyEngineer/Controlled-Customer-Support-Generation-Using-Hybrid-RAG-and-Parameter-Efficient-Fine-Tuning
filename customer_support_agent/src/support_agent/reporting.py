from __future__ import annotations

from pathlib import Path


def write_markdown_report(title: str, body: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")
    return output_path
