import numpy as np
import pyvista as pv
import torch
import argparse
from pathlib import Path
import sys

# 1. Silence VTK Warnings (Texture size, etc.)
#    We access the VTK object directly to turn off global warnings.
try:
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()
except ImportError:
    pass

def _resolve_reconstruction_path(p: Path) -> Path:
    if p.is_dir():
        cand = p / "reconstruction.pt"
        if cand.exists():
            return cand
        raise FileNotFoundError(f"No reconstruction.pt found in directory: {p}")
    return p

def _load_reconstruction_volume(pt_path: Path) -> np.ndarray:
    data = torch.load(pt_path, map_location="cpu")
    if isinstance(data, dict) and "reconstruction" in data:
        vol = data["reconstruction"]
    else:
        vol = data
    return vol.float().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", help="Path to reconstruction.pt")
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--z_step", type=float, default=1.0, help="Z-step in um")
    parser.add_argument("--pixel_size", type=float, default=1.0, help="Pixel size in um")
    args = parser.parse_args()

    # --- Load Data ---
    if args.input_dir:
        input_path = Path(args.input_dir)
    elif args.input:
        input_path = Path(args.input)
    else:
        recons = sorted(list(Path("result").glob("**/reconstruction.pt")))
        if not recons:
            sys.exit("No input provided and no result/reconstruction.pt found.")
        input_path = recons[-1]

    pt_path = _resolve_reconstruction_path(input_path.expanduser().resolve())
    print(f"Loading {pt_path}...")
    
    vol_np = _load_reconstruction_volume(pt_path)
    
    # Clip negative noise (optional)
    vol_np = np.clip(vol_np, a_min=0.0, a_max=None)

    # --- Setup Grid ---
    grid = pv.ImageData()
    grid.dimensions = vol_np.shape
    grid.spacing = (args.pixel_size, args.pixel_size, args.z_step)
    nx, ny, nz = vol_np.shape
    grid.origin = (-(nx-1)/2.0 * args.pixel_size, -(ny-1)/2.0 * args.pixel_size, 0.0)
    grid.point_data["intensity"] = vol_np.ravel(order="F")

    # --- Calculate Dynamic Range ---
    finite_vals = vol_np[np.isfinite(vol_np)]
    if finite_vals.size == 0:
        sys.exit("Volume has no finite values.")
        
    max_density = max(finite_vals.max(), 1e-12)
    
    init_clim_low = 0.0
    init_clim_high = float(np.clip(5.0, 1e-12, max_density))

    # --- Setup Plotter ---
    pl = pv.Plotter(window_size=(1200, 900))
    pl.set_background("white")

    # Add Main Volume
    # We removed add_volume_clip_plane, so this is the ONLY volume in the scene.
    vol_actor = pl.add_volume(
        grid,
        cmap="viridis",
        opacity="linear", 
        clim=(init_clim_low, init_clim_high),
        show_scalar_bar=True,
        scalar_bar_args={"title": "Intensity", "vertical": False},
    )

    # --- Update Callback ---
    state = {"low": init_clim_low, "high": init_clim_high}

    def update_volume(*_):
        low = state["low"]
        high = state["high"]
        
        if high <= low:
            high = low + 1e-5
        
        # 1. Update Color Mapper
        if hasattr(vol_actor, "mapper") and vol_actor.mapper:
            vol_actor.mapper.scalar_range = (low, high)
        
        # 2. Update Opacity
        prop = vol_actor.GetProperty()
        if prop:
            opacity_func = prop.GetScalarOpacity()
            opacity_func.RemoveAllPoints()
            opacity_func.AddPoint(low, 0.0)
            opacity_func.AddPoint(high, 1.0)
            
        # 3. Update Legend Labels
        # Since there is now only one scalar bar (from vol_actor), this works cleanly.
        pl.update_scalar_bar_range([low, high])

    # --- Sliders ---
    pl.add_slider_widget(
        callback=lambda v: (state.update({"low": v}), update_volume()),
        rng=(0, max_density),
        value=init_clim_low,
        title="CLim Low",
        pointa=(0.05, 0.1), pointb=(0.35, 0.1),
        style='modern',
    )
    pl.add_slider_widget(
        callback=lambda v: (state.update({"high": v}), update_volume()),
        rng=(0, max_density),
        value=init_clim_high,
        title="CLim High",
        pointa=(0.40, 0.1), pointb=(0.70, 0.1),
        style='modern',
    )

    # --- Visual Aids (No Clip Plane) ---
    pl.add_mesh(grid.outline(), color="black")
    pl.add_axes()
    pl.show_bounds(grid=True, location='outer', ticks='both')

    # --- View Buttons ---
    def set_view(view_name):
        if view_name == "Top": pl.view_xy()
        elif view_name == "Front": pl.view_zx()
        elif view_name == "Right": pl.view_zy()
        elif view_name == "Iso": pl.view_isometric()

    btns = [("Top", "#FF9999"), ("Front", "#99FF99"), ("Right", "#9999FF"), ("Iso", "#DDDDDD")]
    start_x, start_y = 20, pl.window_size[1] - 50
    for i, (txt, col) in enumerate(btns):
        pl.add_checkbox_button_widget(
            lambda v, t=txt: set_view(t), 
            position=(start_x + i * 50, start_y), 
            size=30, color_on=col, color_off=col
        )

    print("Opening PyVista window...")
    pl.show()

if __name__ == "__main__":
    main()