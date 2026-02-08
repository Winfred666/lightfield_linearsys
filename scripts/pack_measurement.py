import os
import argparse
import glob
import re
import numpy as np
import torch
from PIL import Image

def natural_sort_key(s):
    """Sorts strings containing numbers naturally (1, 2, 10 instead of 1, 10, 2)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def pack_measurement(input_dir, output_name="measurement_packup.pt", crop_box=None):
    """
    Packs a sequence of TIFF images into a 3D PyTorch tensor.
    
    Args:
        input_dir (str): Directory containing '1scan ({d}).tif' images.
        output_name (str): Base name for the output .pt file.
        crop_box (list): Optional [x0, y0, x1, y1] cropping window.
    """
    
    # 1. Find images matching the pattern
    # Pattern: 1scan (*).tif
    pattern = os.path.join(input_dir, "1scan (*).tif")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching pattern '{pattern}'")
        return

    # Sort files naturally (1, 2, ..., 10, ...)
    files = sorted(files, key=natural_sort_key)
    num_images = len(files)
    print(f"Found {num_images} images in {input_dir}.")

    # 2. Determine dimensions from the first image
    img0 = Image.open(files[0])
    W_img, H_img = img0.size  # PIL returns (Width, Height)
    print(f"Original Image Size: W={W_img}, H={H_img}")

    # Convention: H, W maps to W, H = X, Y
    # So X_full = W_img, Y_full = H_img
    
    if crop_box:
        x0, y0, x1, y1 = crop_box
        print(f"Cropping to x[{x0}:{x1}], y[{y0}:{y1}]")
    else:
        x0, y0, x1, y1 = 0, 0, W_img, H_img
    
    out_X = x1 - x0
    out_Y = y1 - y0
    
    # Z-axis logic
    voxels_per_image = 4
    out_Z = num_images * voxels_per_image
    
    print(f"Output Volume Shape (X, Y, Z): ({out_X}, {out_Y}, {out_Z})")
    
    # Allocate volume tensor (X, Y, Z)
    volume = torch.zeros((out_X, out_Y, out_Z), dtype=torch.float32)
    
    for i, fpath in enumerate(files):
        # i is 0-based index of the file in the sorted list
        
        try:
            img = Image.open(fpath)
            arr = np.array(img) # (H, W) -> locally (Y_img, X_img) if we map H->Y, W->X directly? 
            # Wait. np.array(PIL) is (H, W).
            # User wants H,W to W,H = X,Y.
            # So Image H maps to Volume Y. Image W maps to Volume X.
            # Thus array shape (H, W) corresponds to (Y, X).
            # We want (X, Y). So we need to transpose.
            arr = arr.T  # (H, W) -> (W, H) = (X, Y)
            
            # Now arr is (X, Y)
            # Crop
            chunk = arr[x0:x1, y0:y1]
            
            # Normalize signal
            chunk = chunk.astype(np.float32) / float(voxels_per_image)
            
            # Determine Z placement
            # "Image index and Z is reversed"
            # File 0 (First) -> Top Chunk (Highest Z)
            # File N-1 (Last) -> Bottom Chunk (Lowest Z)
            
            z_chunk_idx = num_images - 1 - i
            z_start = z_chunk_idx * voxels_per_image
            z_end = z_start + voxels_per_image
            
            # Copy to volume
            chunk_tensor = torch.from_numpy(chunk)
            
            # Expand to (X, Y, 4) and assign
            volume[:, :, z_start:z_end] = chunk_tensor.unsqueeze(-1).expand(-1, -1, voxels_per_image)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{num_images} -> Z[{z_start}:{z_end}]")
                
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            return

    # Construct output path
    base_name = os.path.splitext(output_name)[0]
    ext = os.path.splitext(output_name)[1]
    
    if crop_box:
        final_name = f"{base_name}_{x0}_{y0}_{x1}_{y1}{ext}"
    else:
        final_name = output_name
        
    output_path = os.path.join(input_dir, final_name)
    
    print(f"Saving volume to {output_path} ...")
    torch.save(volume, output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack TIFF sequence into a 3D PT volume.")
    parser.add_argument("--input_dir", default="data/raw/0um_imgs_crop/0.0um_cropped", help="Input directory containing .tif files")
    parser.add_argument("--output_name", default="measurement_packup.pt", help="Base output filename")
    parser.add_argument("--crop", type=str, help="Crop box x0,y0,x1,y1 (e.g. 1140,830,1652,1342)")
    
    args = parser.parse_args()
    
    crop_box = None
    if args.crop:
        try:
            crop_box = [int(v) for v in args.crop.split(",")]
            if len(crop_box) != 4:
                raise ValueError
        except:
            print("Error: --crop must be four integers separated by commas (x0,y0,x1,y1)")
            exit(1)
            
    pack_measurement(args.input_dir, args.output_name, crop_box)
