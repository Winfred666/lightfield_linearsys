import numpy as np
import pyvista as pv
import torch


def _print_stats(name: str, a: np.ndarray) -> None:
    a = np.asarray(a)
    finite = np.isfinite(a)
    if not np.any(finite):
        print(f"{name}: no finite values (shape={a.shape}, dtype={a.dtype})")
        return
    af = a[finite]
    # Cast to float64 for stable stats when data is float16/float32
    af64 = af.astype(np.float64, copy=False)
    print(
        f"{name} stats (finite only): "
        f"min={np.min(af64):.6g}, max={np.max(af64):.6g}, "
        f"mean={np.mean(af64):.6g}, median={np.median(af64):.6g} "
        f"(shape={a.shape}, dtype={a.dtype})"
    )

# Load volume data
vol_np = torch.load("result/solve_pair/ista/20260122_0934/reconstruction.pt")['reconstruction'].float().cpu().numpy()

_print_stats("volume", vol_np)

# Visualization parameters (assuming these are defined elsewhere)
# z_step: Z-step in μm
# z_step = 27.6  # Example value, replace with actual
z_step = 1.0  # Example value, replace with actual

# PIXEL_SIZE_UM: Pixel size in μm
# PIXEL_SIZE_UM = 27.6  # Example value, replace with actual
PIXEL_SIZE_UM = 1.0  # Example value, replace with actual

# z_start: Z-start in μm
z_start = 0.0  # Example value, replace with actual

# Transpose volume: original (Nz, Ny, Nx) -> (Z, Y, X) to PyVista's (Nx, Ny, Nz) -> (X, Y, Z)
# vol_transposed = np.transpose(vol_np, (2, 1, 0))

print("volume_shape: ", vol_np.shape)
vol_transposed = vol_np # Already (X, Y, Z)

# Create PyVista grid
grid = pv.ImageData()
grid.dimensions = vol_transposed.shape  # (Nx, Ny, Nz)

# Set spacing in μm: XY: PIXEL_SIZE_UM, Z: z_step * 1000
dz_um = z_step / 1.0
grid.spacing = (PIXEL_SIZE_UM, PIXEL_SIZE_UM, dz_um)

# Set origin: XY centered, Z at physical start in μm
nx, ny, nz = vol_transposed.shape
x_min = -(nx - 1) / 2.0 * PIXEL_SIZE_UM
y_min = -(ny - 1) / 2.0 * PIXEL_SIZE_UM
z_origin_um = z_start / 1.0
grid.origin = (x_min, y_min, z_origin_um)

# Add data in Fortran order
grid.point_data["intensity"] = vol_transposed.ravel(order="F")

# Create plotter
pl = pv.Plotter(window_size=(1000, 800))
pl.set_background("white")

# ---- Rendering / legend range control ----
# By default PyVista uses the scalar min/max, which can make the colorbar range huge.
# Set an explicit range here (edit as desired).
# Common choices:
# - (0.0, 1.0) if your reconstruction is normalized
# - (0.0, p99) using percentiles for robustness
use_percentile_range = False
if use_percentile_range:
    finite = np.isfinite(vol_transposed)
    if np.any(finite):
        p1, p99 = np.percentile(vol_transposed[finite], [1, 99])
        clim = (float(p1), float(p99))
    else:
        clim = (0.0, 1.0)
else:
    clim = (0.0, 0.005)

# Add volume with sigmoid opacity for transparency
pl.add_volume(
    grid,
    cmap="viridis",
    opacity="sigmoid",
    clim=clim,
    scalar_bar_args={
        'title': 'Intensity',
        # Keep the legend readable when range is small.
        'fmt': '%.3g',
        'n_labels': 5,
    },
)

# Add outline
outline = grid.outline()
pl.add_mesh(outline, color="black", line_width=2)

# Add axes
axes_actor = pl.add_axes(
    line_width=3,
    labels_off=False,
    viewport=(0, 0, 0.5, 0.5)
)

# Add bounds with Z labels
pl.show_bounds(
    location='outer',
    ticks="both",
    show_xaxis=True,
    show_yaxis=True,
    show_zaxis=True,
    n_zlabels=3,
    xtitle='Y',
    ytitle='X',
    ztitle='Z',
    font_size=12,
    color='black',
    minor_ticks=False,
    use_2d=False
)

# Show the plot
pl.show()