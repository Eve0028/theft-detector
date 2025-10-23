"""
Image Metadata Generation Script
=================================

Generates a CSV file with metadata for all stimulus images.

Metadata includes:
- Image ID
- Type (probe/irrelevant)
- Object name
- View number
- Dimensions
- Mean brightness
- Contrast (RMS)

Usage
-----
::

    python generate_metadata.py
    python generate_metadata.py --image-dir ../images/normalized

"""

import os
import csv
import argparse
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image, ImageStat


def extract_image_info(filename: str) -> Dict[str, str]:
    """
    Extract information from image filename.
    
    Expected format:
    - probe_objectname_viewN.ext
    - irr_objectname_viewN.ext
    
    Parameters
    ----------
    filename : str
        Image filename
        
    Returns
    -------
    dict
        Extracted information
    """
    # Remove extension
    name = Path(filename).stem
    
    # Split by underscores
    parts = name.split('_')
    
    if len(parts) < 3:
        raise ValueError(f"Invalid filename format: {filename}")
    
    # Extract type
    if parts[0] == 'probe':
        img_type = 'probe'
    elif parts[0] == 'irr':
        img_type = 'irrelevant'
    else:
        raise ValueError(f"Unknown image type in filename: {filename}")
    
    # Extract view number (last part)
    view = parts[-1].replace('view', '')
    
    # Object name is everything in between
    object_name = '_'.join(parts[1:-1])
    
    # Generate ID
    img_id = f"{img_type}_{object_name}_{view}"
    
    return {
        'id': img_id,
        'type': img_type,
        'object_name': object_name,
        'view': view,
        'filename': filename
    }


def calculate_image_stats(image_path: str) -> Dict[str, float]:
    """
    Calculate image statistics.
    
    Parameters
    ----------
    image_path : str
        Path to image file
        
    Returns
    -------
    dict
        Image statistics
    """
    # Load image with PIL
    img = Image.open(image_path).convert('RGB')
    
    # Get statistics
    stats = ImageStat.Stat(img)
    
    # Calculate metrics
    mean_brightness = np.mean(stats.mean)
    contrast_rms = np.sqrt(np.mean([s**2 for s in stats.stddev]))
    
    return {
        'width_px': img.size[0],
        'height_px': img.size[1],
        'mean_brightness': round(mean_brightness, 2),
        'contrast_rms': round(contrast_rms, 2)
    }


def generate_metadata(image_dir: str, output_file: str) -> None:
    """
    Generate metadata CSV for all images.
    
    Parameters
    ----------
    image_dir : str
        Directory containing images
    output_file : str
        Output CSV filename
    """
    # Find all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
        image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Collect metadata for all images
    metadata_list = []
    
    for img_path in sorted(image_files):
        print(f"Processing: {img_path.name}")
        
        try:
            # Extract info from filename
            info = extract_image_info(img_path.name)
            
            # Calculate image statistics
            stats = calculate_image_stats(str(img_path))
            
            # Combine
            metadata = {**info, **stats, 'notes': ''}
            metadata_list.append(metadata)
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Write to CSV
    fieldnames = [
        'id', 'type', 'object_name', 'view', 'filename',
        'width_px', 'height_px', 'mean_brightness', 'contrast_rms',
        'notes'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_list)
    
    print(f"\nMetadata saved to: {output_file}")
    print(f"Total images: {len(metadata_list)}")
    
    # Print summary
    probe_count = sum(1 for m in metadata_list if m['type'] == 'probe')
    irrelevant_count = sum(1 for m in metadata_list if m['type'] == 'irrelevant')
    
    print(f"  Probe images: {probe_count}")
    print(f"  Irrelevant images: {irrelevant_count}")
    
    # Print object distribution
    objects = {}
    for m in metadata_list:
        obj = m['object_name']
        objects[obj] = objects.get(obj, 0) + 1
    
    print("\nObject distribution:")
    for obj, count in sorted(objects.items()):
        print(f"  {obj}: {count}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate metadata CSV for stimulus images'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default='../images/normalized',
        help='Directory containing images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../images/image_metadata.csv',
        help='Output CSV filename'
    )
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    script_dir = Path(__file__).parent
    image_dir = script_dir / args.image_dir
    output_file = script_dir / args.output
    
    # Check if image directory exists
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        print("Please run normalize_images.py first to create normalized images.")
        return
    
    # Generate metadata
    generate_metadata(
        image_dir=str(image_dir),
        output_file=str(output_file)
    )


if __name__ == '__main__':
    main()

