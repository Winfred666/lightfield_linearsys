from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence
import re

@dataclass(frozen=True)
class RawPair:
    idx: int
    vol_path: Path
    img_path: Path

def _infer_idx_regex_from_examples(names: Sequence[str]) -> Optional[str]:
    """
    Infer a regex with one named capture group `idx` from 2-3 filenames.
    It finds digit runs that vary across examples and treats stable text as literal.
    """
    if len(names) < 2:
        return None

    # Tokenize each name into alternating [non-digits, digits, non-digits, ...]

    tokenized = [re.findall(r"\d+|\D+", n) for n in names]
    lengths = {len(t) for t in tokenized}
    if len(lengths) != 1:
        return None  # incompatible structures

    n_tokens = lengths.pop()
    varying_digit_positions = []

    for i in range(n_tokens):
        col = [t[i] for t in tokenized]
        all_digit = all(c.isdigit() for c in col)
        if all_digit and len(set(col)) > 1:
            varying_digit_positions.append(i)

    if not varying_digit_positions:
        return None

    # Prefer the last varying digit token as index (common for *_ID_123.ext, (...123).tif)
    idx_pos = varying_digit_positions[-1]

    parts: list[str] = ["^"]
    for i in range(n_tokens):
        ref = tokenized[0][i]
        if i == idx_pos:
            parts.append(r"(?P<idx>\d+)")
        else:
            parts.append(re.escape(ref))
    parts.append("$")
    return "".join(parts)


def _extract_idx_with_regex(name: str, pattern: str) -> Optional[int]:

    m = re.match(pattern, name)
    if not m:
        return None
    try:
        return int(m.group("idx"))
    except (ValueError, IndexError):
        return None


def find_raw_pairs(
    input_dir: str | Path,
    img_dir: str | Path,
    *,
    vol_glob: str = "*.pt",
    img_glob: str = "*.tif",
    max_pairs: Optional[int] = None,
    infer_samples: int = 3,
) -> List[RawPair]:
    """
    Robustly pair raw volumes and images by auto-inferring index patterns from filenames.

    Strategy:
    1) Collect candidate volume/image files.
    2) Infer index regex from 2-3 sample filenames per side.
    3) Fallback to 'last digit run in stem' if inference fails.
    4) Pair by shared integer index.

    Notes:
    - Keeps original behavior of returning [] if dirs do not exist.
    - Sorted by idx ascending.
    """

    input_dir_p = Path(input_dir)
    img_dir_p = Path(img_dir)

    if not input_dir_p.exists() or not img_dir_p.exists():
        return []

    vol_files = sorted(input_dir_p.glob(vol_glob))
    img_files = sorted(img_dir_p.glob(img_glob))
    if not vol_files or not img_files:
        return []

    def build_idx_map(files: Sequence[Path]) -> dict[int, Path]:
        # Try infer from up to 2-3 examples
        sample_names = [p.name for p in files[: max(2, min(infer_samples, len(files)))]]
        pattern = _infer_idx_regex_from_examples(sample_names)

        idx_map: dict[int, Path] = {}
        for p in files:
            idx: Optional[int] = None
            if pattern is not None:
                idx = _extract_idx_with_regex(p.name, pattern)

            # Fallback: last digit run in stem
            if idx is None:
                m = re.findall(r"\d+", p.stem)
                if m:
                    idx = int(m[-1])

            if idx is not None and idx not in idx_map:
                idx_map[idx] = p
        return idx_map

    vol_by_idx = build_idx_map(vol_files)
    img_by_idx = build_idx_map(img_files)

    common_idx = sorted(set(vol_by_idx.keys()) & set(img_by_idx.keys()))
    pairs = [
        RawPair(idx=i, vol_path=vol_by_idx[i], img_path=img_by_idx[i])
        for i in common_idx
    ]

    if max_pairs is not None:
        pairs = pairs[: int(max_pairs)]

    return pairs


def to_driver_file_dicts(pairs: Sequence[RawPair]) -> list[dict]:
    """Compatibility helper for drivers expecting dict-based file descriptors."""
    return [
        {"type": "raw", "vol_path": p.vol_path, "img_path": p.img_path, "id": p.idx}
        for p in pairs
    ]