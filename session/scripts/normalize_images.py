"""
Image Normalization Script
===========================

Normalizes stimulus images for the P300-CIT experiment.

Operations performed:
- Resize images to consistent dimensions
- Normalize brightness and contrast
- Convert to standard format (RGB, PNG)
- Save normalized versions

Usage
-----
::

    python normalize_images.py
    python normalize_images.py --input-dir ../images --output-dir ../images/normalized

"""

import os
import argparse
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageStat, ImageEnhance


class ImageNormalizer:
    """
    Normalize images for experimental stimuli.
    
    Parameters
    ----------
    target_size : tuple of int, optional
        Target size (width, height) in pixels. If None, uses original size.
    target_brightness : float, optional
        Target mean brightness (0-255). If None, no brightness adjustment.
    target_contrast : float, optional
        Target contrast (RMS). If None, no contrast adjustment.
    maintain_aspect : bool, optional
        Whether to maintain aspect ratio (default: True)
    """
    
    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = (800, 600),
        target_brightness: Optional[float] = 128.0,
        target_contrast: Optional[float] = 50.0,
        maintain_aspect: bool = True
    ):
        self.target_size = target_size
        self.target_brightness = target_brightness
        self.target_contrast = target_contrast
        self.maintain_aspect = maintain_aspect
    
    def normalize_image(self, image_path: str, output_path: str) -> dict:
        """
        Normalize a single image.
        
        Parameters
        ----------
        image_path : str
            Path to input image
        output_path : str
            Path to save normalized image
            
        Returns
        -------
        dict
            Statistics about the normalized image
        """
        # Load image with PIL
        img_pil = Image.open(image_path).convert('RGB')
        
        # Get original statistics
        orig_stats = self._get_image_stats(img_pil)
        
        # Resize
        if self.target_size is not None:
            img_pil = self._resize_image(img_pil)
        
        # Normalize brightness
        if self.target_brightness is not None:
            img_pil = self._normalize_brightness(img_pil)
        
        # Normalize contrast
        if self.target_contrast is not None:
            img_pil = self._normalize_contrast(img_pil)
        
        # Get final statistics
        final_stats = self._get_image_stats(img_pil)
        
        # Save normalized image
        img_pil.save(output_path, 'PNG', quality=95)
        
        return {
            'original': orig_stats,
            'normalized': final_stats
        }
    
    def _resize_image(self, img: Image.Image) -> Image.Image:
        """Resize image to target size."""
        if self.maintain_aspect:
            # Calculate size maintaining aspect ratio
            img.thumbnail(self.target_size, Image.Resampling.LANCZOS)
            
            # Create blank canvas of target size
            canvas = Image.new('RGB', self.target_size, (128, 128, 128))
            
            # Paste resized image in center
            offset = (
                (self.target_size[0] - img.size[0]) // 2,
                (self.target_size[1] - img.size[1]) // 2
            )
            canvas.paste(img, offset)
            
            return canvas
        else:
            return img.resize(self.target_size, Image.Resampling.LANCZOS)
    
    def _normalize_brightness(self, img: Image.Image) -> Image.Image:
        """Normalize image brightness to target value."""
        # Calculate current mean brightness
        stats = ImageStat.Stat(img)
        current_brightness = np.mean(stats.mean)
        
        # Calculate adjustment factor
        if current_brightness > 0:
            factor = self.target_brightness / current_brightness
            
            # Apply brightness adjustment
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
        
        return img
    
    def _normalize_contrast(self, img: Image.Image) -> Image.Image:
        """Normalize image contrast to target RMS value."""
        # Calculate current contrast (RMS of standard deviations)
        stats = ImageStat.Stat(img)
        current_contrast = np.sqrt(np.mean([s**2 for s in stats.stddev]))
        
        # Calculate adjustment factor
        if current_contrast > 0:
            factor = self.target_contrast / current_contrast
            factor = np.clip(factor, 0.5, 2.0)  # Limit extreme adjustments
            
            # Apply contrast adjustment
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)
        
        return img
    
    def _get_image_stats(self, img: Image.Image) -> dict:
        """Get image statistics."""
        stats = ImageStat.Stat(img)
        
        return {
            'width': img.size[0],
            'height': img.size[1],
            'mean_brightness': np.mean(stats.mean),
            'contrast_rms': np.sqrt(np.mean([s**2 for s in stats.stddev]))
        }


def process_directory(
    input_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (800, 600),
    target_brightness: float = 128.0,
    target_contrast: float = 50.0
) -> None:
    """
    Process all images in a directory.
    
    Parameters
    ----------
    input_dir : str
        Input directory containing raw images
    output_dir : str
        Output directory for normalized images
    target_size : tuple of int
        Target image size (width, height)
    target_brightness : float
        Target mean brightness
    target_contrast : float
        Target contrast RMS
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize normalizer
    normalizer = ImageNormalizer(
        target_size=target_size,
        target_brightness=target_brightness,
        target_contrast=target_contrast,
        maintain_aspect=True
    )
    
    # Find all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    # Process each image
    print(f"Found {len(image_files)} images to process")
    print(f"Target size: {target_size}")
    print(f"Target brightness: {target_brightness}")
    print(f"Target contrast: {target_contrast}")
    print()
    
    for i, img_path in enumerate(sorted(image_files)):
        print(f"Processing [{i+1}/{len(image_files)}]: {img_path.name}")
        
        # Generate output filename (convert to PNG)
        output_filename = img_path.stem + '.png'
        output_path = os.path.join(output_dir, output_filename)
        
        # Normalize image
        stats = normalizer.normalize_image(str(img_path), output_path)
        
        # Print statistics
        print(f"  Original: {stats['original']['width']}x{stats['original']['height']}, "
              f"brightness={stats['original']['mean_brightness']:.1f}, "
              f"contrast={stats['original']['contrast_rms']:.1f}")
        print(f"  Normalized: {stats['normalized']['width']}x{stats['normalized']['height']}, "
              f"brightness={stats['normalized']['mean_brightness']:.1f}, "
              f"contrast={stats['normalized']['contrast_rms']:.1f}")
        print()
    
    print(f"All images processed! Saved to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Normalize stimulus images for P300-CIT experiment'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='../images',
        help='Input directory containing raw images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../images/normalized',
        help='Output directory for normalized images'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=800,
        help='Target width in pixels'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=600,
        help='Target height in pixels'
    )
    parser.add_argument(
        '--brightness',
        type=float,
        default=128.0,
        help='Target mean brightness (0-255)'
    )
    parser.add_argument(
        '--contrast',
        type=float,
        default=50.0,
        help='Target contrast (RMS of standard deviation)'
    )
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input_dir
    output_dir = script_dir / args.output_dir
    
    # Process images
    process_directory(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        target_size=(args.width, args.height),
        target_brightness=args.brightness,
        target_contrast=args.contrast
    )


if __name__ == '__main__':
    main()

