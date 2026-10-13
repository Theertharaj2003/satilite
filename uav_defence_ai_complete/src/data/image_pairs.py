from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImagePair:
    eo: Path
    ir: Path
    key: str


def _index(folder: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            result[path.stem] = path
    return result


def pair_images(eo_dir: str | Path, ir_dir: str | Path) -> Iterator[ImagePair]:
    eo = _index(Path(eo_dir))
    ir = _index(Path(ir_dir))
    for key in sorted(set(eo) & set(ir)):
        yield ImagePair(eo=eo[key], ir=ir[key], key=key)
