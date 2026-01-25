import argparse
from pathlib import Path

import numpy as np
import torch
from skimage import measure


def export_volume_to_obj(volume, output_path, iso_value=0.5, flip_axes=False):
    """
    Converts a 3D volume to a Wavefront OBJ mesh using marching cubes.

    Args:
        volume (np.ndarray): 3D numpy array of the volume data.
        output_path (str or Path): Path to save the output OBJ file.
        iso_value (float): Iso-value for the surface extraction.
        flip_axes (bool): If True, flips the Y and Z axes, which is often
                          required for visualization in standard 3D viewers.
    """
    if not isinstance(volume, np.ndarray):
        raise TypeError("Volume must be a numpy array.")

    if volume.ndim != 3:
        raise ValueError("Volume must be 3-dimensional.")

    # Marching cubes algorithm to get vertices and faces
    try:
        verts, faces, _, _ = measure.marching_cubes(volume, level=iso_value)
    except Exception as e:
        print(f"Marching cubes failed: {e}")
        try:
            verts, faces = measure.marching_cubes_classic(volume, level=iso_value)
        except AttributeError:
            print("`marching_cubes_classic` not available. Your scikit-image version may be too old or too new.")
            return
        except Exception as e_classic:
            print(f"Classic marching cubes also failed: {e_classic}")
            return

    if flip_axes:
        # (X, Y, Z) -> (X, -Z, Y)
        verts = verts[:, [0, 2, 1]]
        verts[:, 1] *= -1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to OBJ file
    with open(output_path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    print(f"Successfully exported mesh to {output_path} with {len(verts)} vertices and {len(faces)} faces.")


def _to_numpy_volume(reconstruction):
    if isinstance(reconstruction, np.ndarray):
        vol = reconstruction
    elif torch.is_tensor(reconstruction):
        vol = reconstruction.detach().cpu().numpy()
    else:
        raise TypeError(f"Unsupported reconstruction type: {type(reconstruction)}")

    # Allow (1, D, H, W) or (D, H, W)
    if vol.ndim == 4 and vol.shape[0] == 1:
        vol = vol[0]
    if vol.ndim != 3:
        raise ValueError(f"'reconstruction' must be 3D (or 1x3D). Got shape: {vol.shape}")

    # Ensure float for marching cubes
    return vol.astype(np.float32, copy=False)


def main():
    parser = argparse.ArgumentParser(description="Convert a .pt file with 'reconstruction' volume to an OBJ mesh.")
    parser.add_argument("pt_path", type=str, help="Path to .pt file containing a 'reconstruction' field.")
    parser.add_argument("--iso_value", type=float, default=0.5, help="Iso-value for marching cubes.")
    parser.add_argument("--flip_axes", action="store_true", help="Flip Y/Z axes for common viewer conventions.")
    args = parser.parse_args()

    pt_path = Path(args.pt_path)
    if not pt_path.is_file():
        raise FileNotFoundError(f"Not found: {pt_path}")

    data = torch.load(pt_path, map_location="cpu")
    if not isinstance(data, dict) or "reconstruction" not in data:
        raise KeyError(f"{pt_path} must be a dict-like object containing key 'reconstruction'.")

    volume = _to_numpy_volume(data["reconstruction"])

    out_path = pt_path.parent / f"reconstruction_iso_{args.iso_value}.obj"
    export_volume_to_obj(volume, out_path, iso_value=args.iso_value, flip_axes=args.flip_axes)


if __name__ == "__main__":
    main()
