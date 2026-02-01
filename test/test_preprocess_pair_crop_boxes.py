import numpy as np
import torch
import pytest


def test_preprocess_one_pair_manual_crop_boxes(monkeypatch):
    """Manual crop boxes should be applied before any new processing.

    We patch readers to avoid IO and scale_volume to identity so the crop result
    is deterministic.
    """
    from LF_linearsys.io import preprocess_pair as pp

    # Synthetic inputs
    # vol: (X,Y,Z) = (6,5,4)
    vol = torch.arange(6 * 5 * 4, dtype=torch.float32).reshape(6, 5, 4)
    # img: (Y,X) = (20,24)
    img = np.arange(20 * 24, dtype=np.float32).reshape(20, 24)

    monkeypatch.setattr(pp, "read_volume", lambda p: vol.clone())
    monkeypatch.setattr(pp, "read_image", lambda p: img.copy())
    monkeypatch.setattr(pp, "scale_volume", lambda v, scale_factor: v)

    # Crop boxes must satisfy the strict scaling rule enforced by preprocess_pair:
    #   (dx,dy)_b == round((dx,dy)_A * scale_factor)
    # Here scale_factor=2.0, so b crop must be 2x A crop in x/y.
    crop_box_A = (1, 1, 1, 5, 4, 3)  # dx=4, dy=3, dz=2
    crop_box_b = (2, 3, 10, 9)       # dx=8, dy=6  (x0,y0,x1,y1)

    A_cpu, b_cpu = pp.preprocess_one_pair(
        vol_path=torch.tensor(0),  # dummy
        img_path=torch.tensor(0),
        downsampling_rate=0.5,
        scale_factor=2.0,
        crop_box_b=crop_box_b,
        crop_box_A=crop_box_A,
        device=torch.device("cpu"),
    )

    # scale_volume is identity, so the *manual* crop should be present inside the output.
    # The function may still apply its own center crops later, so we assert that the
    # returned tensors equal a centered sub-crop of our manual-cropped tensors.
    manual_A = vol[crop_box_A[0] : crop_box_A[3], crop_box_A[1] : crop_box_A[4], crop_box_A[2] : crop_box_A[5]]
    manual_b = torch.from_numpy(img[crop_box_b[1] : crop_box_b[3], crop_box_b[0] : crop_box_b[2]]).float()

    # The function downsamples b using bilinear interpolation.
    manual_b_in = manual_b.unsqueeze(0).unsqueeze(0)
    manual_b = (
        torch.nn.functional.interpolate(manual_b_in, scale_factor=0.5, mode="bilinear", align_corners=False)
        .squeeze(0)
        .squeeze(0)
    )

    assert A_cpu.shape[2] == manual_A.shape[2]  # Z preserved in this test
    assert b_cpu.shape[0] <= manual_b.shape[0]
    assert b_cpu.shape[1] <= manual_b.shape[1]

    # Center-crop manual tensors to the returned shape.
    xa0 = (manual_A.shape[0] - A_cpu.shape[0]) // 2
    ya0 = (manual_A.shape[1] - A_cpu.shape[1]) // 2
    expected_A = manual_A[xa0 : xa0 + A_cpu.shape[0], ya0 : ya0 + A_cpu.shape[1], :]

    yb0 = (manual_b.shape[0] - b_cpu.shape[0]) // 2
    xb0 = (manual_b.shape[1] - b_cpu.shape[1]) // 2
    expected_b = manual_b[yb0 : yb0 + b_cpu.shape[0], xb0 : xb0 + b_cpu.shape[1]]

    assert torch.allclose(A_cpu, expected_A)
    assert torch.allclose(b_cpu, expected_b)


def test_preprocess_one_pair_manual_crop_mismatch_raises(monkeypatch):
    from LF_linearsys.io import preprocess_pair as pp

    vol = torch.zeros((6, 5, 4), dtype=torch.float32)
    img = np.zeros((20, 24), dtype=np.float32)

    monkeypatch.setattr(pp, "read_volume", lambda p: vol)
    monkeypatch.setattr(pp, "read_image", lambda p: img)
    monkeypatch.setattr(pp, "scale_volume", lambda v, scale_factor: v)

    # scale_factor=2.0; A crop dx=4,dy=3 expects b crop dx=8,dy=6.
    # Here we intentionally mismatch dx to ensure strict validation raises.
    crop_box_A = (1, 1, 1, 5, 4, 3)  # dx=4, dy=3
    crop_box_b = (2, 3, 11, 9)       # dx=9 (mismatch), dy=6

    with pytest.raises(ValueError, match=r"crop boxes mismatch"):
        pp.preprocess_one_pair(
            vol_path=torch.tensor(0),
            img_path=torch.tensor(0),
            downsampling_rate=0.5,
            scale_factor=2.0,
            crop_box_b=crop_box_b,
            crop_box_A=crop_box_A,
            device=torch.device("cpu"),
        )
