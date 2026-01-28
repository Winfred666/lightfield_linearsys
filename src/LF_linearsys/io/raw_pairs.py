from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

@dataclass(frozen=True)
class RawPair:
    idx: int
    vol_path: Path
    img_path: Path

def find_raw_pairs(
    input_dir: str | Path,
    img_dir: str | Path,
    *,
    vol_glob: str = "Interp_Vol_ID_*.pt",
    img_name_template: str = "1scan ({idx}).tif",
    max_pairs: Optional[int] = None,
) -> List[RawPair]:
    """
    Finds and pairs raw volumes and images based on their indices.
    
    Expected naming conventions:
    - Volume: Interp_Vol_ID_{idx}.pt inside input_dir
    - Image: 1scan ({idx}).tif inside img_dir
    """
    input_dir_p = Path(input_dir)
    img_dir_p = Path(img_dir)
    
    if not input_dir_p.exists() or not img_dir_p.exists():
        return []

    vol_files = sorted(list(input_dir_p.glob(vol_glob)))
    pairs = []
    
    for vp in vol_files:
        try:
            # Expected format: Interp_Vol_ID_{idx}.pt
            idx = int(vp.stem.split('_')[-1])
            img_path = img_dir_p / img_name_template.format(idx=idx)
            
            if img_path.exists():
                pairs.append(RawPair(idx=idx, vol_path=vp, img_path=img_path))
        except (ValueError, IndexError):
            continue
            
    # Sort by index
    pairs.sort(key=lambda p: p.idx)

    if max_pairs is not None:
        pairs = pairs[: int(max_pairs)]

    return pairs


def to_driver_file_dicts(pairs: Sequence[RawPair]) -> list[dict]:
    """Compatibility helper for drivers expecting dict-based file descriptors."""
    return [
        {"type": "raw", "vol_path": p.vol_path, "img_path": p.img_path, "id": p.idx}
        for p in pairs
    ]