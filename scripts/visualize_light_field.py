import numpy as np
import pyvista as pv
import torch

# Load volume data
vol_np = torch.load("result/solve_pair/ista/20260122_0252/reconstruction.pt").cpu().numpy()

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

# Add volume with sigmoid opacity for transparency
pl.add_volume(
    grid,
    cmap="viridis",
    opacity="sigmoid",
    scalar_bar_args={'title': 'Intensity'}
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