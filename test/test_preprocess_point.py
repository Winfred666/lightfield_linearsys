import os
from pathlib import Path

import h5py
import numpy as np
import torch


def _write_pair_h5(path: Path, A: np.ndarray, b: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("A", data=A)
        f.create_dataset("b", data=b)


def test_preprocess_point_packs_pairs_correctly():
    """End-to-end check that preprocess_point packs the same pixel across files.

    We write pair_1..pair_3 with known per-(x,y) patterns, run preprocess, and
    verify that points_batch_0000.pt contains:
      A.shape == (X*Y, num_pairs, Z)
      b.shape == (X*Y, num_pairs)
    and for random points i and each measurement m:
      A[i,m,:] equals original A_m[x,y,:]
      b[i,m]   equals original b_m[y,x]
    """
    # Import here so the test doesn't require the module at collection time
    from LF_linearsys.io.preprocess_point import preprocess_points_optimized

    out_dir = Path("result/solve/preprocess_point_test")
    data_dir = out_dir / "pairs"
    output_dir = out_dir  # user requested output in same folder

    # Clean/ensure directories
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Small synthetic sizes
    X, Y, Z = 3, 2, 4
    num_pairs = 3

    # Build deterministic A_m and b_m
    # A_m[x,y,z] = (m+1)*1000 + x*100 + y*10 + z
    # b_m[y,x]   = (m+1)*1000 + y*10 + x
    A_list: list[np.ndarray] = []
    b_list: list[np.ndarray] = []
    for m in range(num_pairs):
        A_m = np.zeros((X, Y, Z), dtype=np.float32)
        b_m = np.zeros((Y, X), dtype=np.float32)
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    A_m[x, y, z] = (m + 1) * 1000 + x * 100 + y * 10 + z
                b_m[y, x] = (m + 1) * 1000 + y * 10 + x
        A_list.append(A_m)
        b_list.append(b_m)

    # Write pair_1..pair_3 (note: preprocess_point natural-sorts by the numeric suffix)
    for m in range(num_pairs):
        _write_pair_h5(data_dir / f"pair_{m+1}.h5", A_list[m], b_list[m])

    # Run preprocess: ensure single batch by setting batch_size >= X*Y
    preprocess_points_optimized(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        batch_size=10_000,
        batches_per_pass=1,
        limit_batches=1,
        save_dtype="float32",
        log_every_pairs=1,
        sync_timing=False,
    )

    pt_path = output_dir / "points_batch_0000.pt"
    assert pt_path.exists(), f"Expected output {pt_path} to exist"

    data = torch.load(pt_path, map_location="cpu")
    A = data["A"]
    b = data["b"]
    coords = data["coords"]

    assert A.shape == (X * Y, num_pairs, Z)
    assert b.shape == (X * Y, num_pairs)
    assert coords.shape == (X * Y, 2)

    # Check a handful of points (including first and last)
    idxs = [0, 1, int(X * Y) - 1]
    for i in idxs:
        x = int(coords[i, 0].item())
        y = int(coords[i, 1].item())
        for m in range(num_pairs):
            expected_A = torch.tensor(A_list[m][x, y, :], dtype=torch.float32)
            expected_b = float(b_list[m][y, x])
            assert torch.allclose(A[i, m, :].float(), expected_A, atol=0, rtol=0)
            assert float(b[i, m].item()) == expected_b


def test_preprocess_point_raw_mode(monkeypatch):
    """Test raw-mode packing without actual pt/tif IO by patching preprocess_one_pair.

    We stub preprocess_one_pair to return deterministic tensors for each idx.
    preprocess_points_from_raw should stack those into points_batch_0000.pt.
    """
    from LF_linearsys.io import preprocess_point as pp
    import LF_linearsys.io.preprocess_pair as preprocess_pair

    out_dir = Path("result/solve/preprocess_point_test_raw")
    input_dir = out_dir / "raw_vols"
    img_dir = out_dir / "raw_imgs"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy filenames so the glob/exists checks pass.
    indices = [1, 2, 3]
    for idx in indices:
        (input_dir / f"Interp_Vol_ID_{idx}.pt").write_text("dummy")
        (img_dir / f"1scan ({idx}).tif").write_text("dummy")

    X, Y, Z = 2, 2, 3
    num_pairs = len(indices)

    def fake_preprocess_one_pair(*, vol_path, img_path, downsampling_rate, scale_factor=8.0, device=None):
        # Encode idx from filename
        idx = int(Path(vol_path).stem.split('_')[-1])
        m = indices.index(idx)  # 0..2
        A = torch.zeros(X, Y, Z, dtype=torch.float32)
        b = torch.zeros(Y, X, dtype=torch.float32)
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    A[x, y, z] = (m + 1) * 1000 + x * 100 + y * 10 + z
                b[y, x] = (m + 1) * 1000 + y * 10 + x
        return A, b

    monkeypatch.setattr(preprocess_pair, "preprocess_one_pair", fake_preprocess_one_pair)

    written = pp.preprocess_points_from_raw(
        input_dir=str(input_dir),
        img_dir=str(img_dir),
        output_dir=str(out_dir),
        downsampling_rate=0.5,
        scale_factor=8.0,
        batch_size=10_000,
        batches_per_pass=1,
        limit_batches=1,
        save_dtype="float32",
        log_every_pairs=1,
    )

    pt_path = out_dir / "points_batch_0000.pt"
    assert pt_path in written or pt_path.exists()

    data = torch.load(pt_path, map_location="cpu")
    A = data["A"]
    b = data["b"]
    coords = data["coords"]
    assert A.shape == (X * Y, num_pairs, Z)
    assert b.shape == (X * Y, num_pairs)

    # Check one point is stacked correctly across all pairs
    i = 0
    x = int(coords[i, 0].item())
    y = int(coords[i, 1].item())
    for m in range(num_pairs):
        expected_A = torch.tensor([(m + 1) * 1000 + x * 100 + y * 10 + z for z in range(Z)], dtype=torch.float32)
        expected_b = float((m + 1) * 1000 + y * 10 + x)
        assert torch.allclose(A[i, m, :].float(), expected_A, atol=0, rtol=0)
        assert float(b[i, m].item()) == expected_b
