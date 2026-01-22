import numpy as np
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
        # Check if it's due to scikit-image version
        try:
            # For older scikit-image versions, the function might not have a `level` argument
            # or have a different API. This is a common point of failure.
            # Let's try the older API signature if it exists.
            verts, faces = measure.marching_cubes_classic(volume, level=iso_value)
        except AttributeError:
             print("`marching_cubes_classic` not available. Your scikit-image version may be too old or too new.")
             return # Abort if we can't run marching cubes
        except Exception as e_classic:
            print(f"Classic marching cubes also failed: {e_classic}")
            return
            
    if flip_axes:
        # Swap Y and Z coordinates and invert the new Z
        # (X, Y, Z) -> (X, -Z, Y)
        verts = verts[:, [0, 2, 1]]
        verts[:, 1] *= -1

    # Write to OBJ file
    with open(output_path, 'w') as f:
        # Write vertices
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
        # Write faces
        # OBJ format uses 1-based indexing for vertices
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"Successfully exported mesh to {output_path} with {len(verts)} vertices and {len(faces)} faces.")

