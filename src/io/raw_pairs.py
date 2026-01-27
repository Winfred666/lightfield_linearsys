from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RawPair:
    idx: int
    vol_path: Path
    img_path: Path

def find_raw_pairs(input_dir: str | Path, img_dir: str | Path) -> List[RawPair]:
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

    vol_files = sorted(list(input_dir_p.glob("Interp_Vol_ID_*.pt")))
    pairs = []
    
    for vp in vol_files:
        try:
            # Expected format: Interp_Vol_ID_{idx}.pt
            idx = int(vp.stem.split('_')[-1])
            img_path = img_dir_p / f"1scan ({idx}).tif"
            
            if img_path.exists():
                pairs.append(RawPair(idx=idx, vol_path=vp, img_path=img_path))
        except (ValueError, IndexError):
            continue
            
    # Sort by index
    pairs.sort(key=lambda p: p.idx)
    return pairs