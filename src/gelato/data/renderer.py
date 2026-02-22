import subprocess
from pathlib import Path
import logging
import os
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class Renderer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render_abc_to_svg(self, abc_file: Path) -> list[Path]:
        """
        Renders ABC file to SVG(s) using abcm2ps.
        Returns a list of generated SVG paths.
        """
        # abcm2ps -g -E -O <output_base> <input_abc>
        # -g: SVG output
        # -E: EPS output (compatible) - actually -g is enough usually
        # doc says: -g produces SVG.
        
        base_name = abc_file.stem
        output_prefix = self.output_dir / base_name
        
        cmd = ["abcm2ps", "-g", "-O", str(output_prefix), str(abc_file)]
        
        logger.info(f"Rendering {abc_file} to SVG...")
        try:
            # Add timeout to prevent hanging on problematic files
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired as e:
            logger.error(f"abcm2ps timed out processing {abc_file}")
            raise RuntimeError(f"Timed out rendering {abc_file}") from e
        except subprocess.CalledProcessError as e:
            logger.error(f"abcm2ps failed: {e.stderr}")
            raise RuntimeError(f"Failed to render {abc_file}") from e

        # abcm2ps produces output001.svg, output002.svg, etc.
        # Find them.
        svgs = sorted(list(self.output_dir.glob(f"{base_name}*.svg")))
        return svgs

    def convert_svg_to_png(self, svg_file: Path, width: int = 1024) -> Path:
        """
        Converts SVG to PNG using ImageMagick (convert).
        Adjusts density to achieve target width approx.
        """
        png_path = svg_file.with_suffix('.png')
        
        # Use rsvg-convert
        # rsvg-convert -d 300 -p 300 -w <width> -f png -o output.png input.svg
        # We can specify width directly, which is nice.
        # But we need to maintain aspect ratio? rsvg-convert does that by default if only w is specified.
        
        # Note: -d and -p are DPI. Default is usually 90.
        # If we specify width, we don't strictly need DPI unless for text rendering size?
        # Let's just specify width.
        
        cmd = ["rsvg-convert", "-w", str(width), "-f", "png", "-o", str(png_path), str(svg_file), "--background-color", "white"]
        
        logger.info(f"Converting {svg_file} to PNG using rsvg-convert...")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired as e:
            logger.error(f"rsvg-convert timed out processing {svg_file}")
            raise RuntimeError(f"Timed out converting {svg_file}") from e
        except subprocess.CalledProcessError as e:
            logger.error(f"rsvg-convert failed: {e.stderr}")
            raise RuntimeError(f"Failed to convert {svg_file} to PNG") from e
            
        return png_path

    def process_image_for_model(self, image_path: Path) -> list[Image.Image]:
        """
        Loads image, standardizes width to 512, pads to height 2048 (4 * 512),
        or splits if taller.
        returns list of processed PIL images composed of vertically stacked patches?
        
        Plan says:
        1. Crops into vertical segments with 1:4 AR.
        2. Resizes to 512w.
        3. Splits into N patches (4).
        
        This effectively means we want chunks of the score that are height = 4 * Width.
        And then resize so Width=512, Height=2048.
         Then split into 4 512x512 patches.
        """
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        
        # We want exactly ONE segment per input image (max 512x2048).
        # We need to scale the image proportionally so it fits within 512x2048.
        target_w = 512
        target_h_max = 4 * target_w # 2048
        
        # Calculate scaling to fit within 512x2048 while maintaining aspect ratio
        scale_w = target_w / w
        scale_h = target_h_max / h
        scale = min(scale_w, scale_h) # Fit within both bounds
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create a blank white canvas of exactly 512x2048
        canvas = Image.new("RGB", (target_w, target_h_max), (255, 255, 255))
        
        # Paste the resized image onto the top-left (or center) of the canvas
        # Top-left alignment is standard for reading order
        canvas.paste(img_resized, (0, 0))
        
        # Return exactly 1 segment
        return [canvas]

    def slice_segment(self, segment: Image.Image, num_patches: int = 4) -> list[Image.Image]:
        """
        Splits a 512x2048 segment into 4 512x512 patches.
        """
        w, h = segment.size
        # Assuming w=512, h=2048
        patch_h = h // num_patches
        
        patches = []
        for i in range(num_patches):
            box = (0, i * patch_h, w, (i + 1) * patch_h)
            patch = segment.crop(box)
            patches.append(patch)
        return patches
