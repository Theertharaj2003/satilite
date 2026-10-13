from __future__ import annotations

from pathlib import Path


def build_index(source_dir: str | Path = "docs", output: str | Path = "outputs/reports/rag_index.txt") -> Path:
    source_dir = Path(source_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            rows.append(f"SOURCE: {path}\n{path.read_text(encoding='utf-8', errors='ignore')}\n\n")
    output.write_text("\n".join(rows), encoding="utf-8")
    return output
