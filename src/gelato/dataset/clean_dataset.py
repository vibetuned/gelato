import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Clean up unmatched ABC and rendered (img/svg) files.")
    parser.add_argument("--abc_dir", type=str, default="data/output_abc", help="Directory containing ABC files")
    parser.add_argument("--img_dir", type=str, default="data/output_imgs", help="Directory containing PNG files")
    parser.add_argument("--svg_dir", type=str, default="data/output_svgs", help="Directory containing SVG files (optional)")
    parser.add_argument("--dry_run", action="store_true", help="Print what would be deleted without actually deleting")
    args = parser.parse_args()

    abc_path = Path(args.abc_dir)
    img_path = Path(args.img_dir)
    svg_path = Path(args.svg_dir)

    abc_files = list(abc_path.rglob("*.abc")) if abc_path.exists() else []
    abc_stems = {f.stem for f in abc_files}
    
    img_files = list(img_path.rglob("*.png")) if img_path.exists() else []
    img_stems = {f.stem for f in img_files}

    svg_files = list(svg_path.rglob("*.svg")) if svg_path.exists() else []
    
    # Missing sets
    stems_missing_imgs = abc_stems - img_stems
    stems_missing_abcs = img_stems - abc_stems
    
    delete_count = 0
    
    # Perform deletes for ABCs
    for abc_file in abc_files:
        if abc_file.stem in stems_missing_imgs:
            if args.dry_run:
                logger.info(f"[Dry Run] Would delete {abc_file}")
            else:
                abc_file.unlink()
                logger.info(f"Deleted {abc_file}")
            delete_count += 1
            
    # Perform deletes for Images
    for img_file in img_files:
        if img_file.stem in stems_missing_abcs:
            if args.dry_run:
                logger.info(f"[Dry Run] Would delete {img_file}")
            else:
                img_file.unlink()
                logger.info(f"Deleted {img_file}")
            delete_count += 1
            
    # Perform deletes for SVGs
    for svg_file in svg_files:
        if svg_file.stem not in abc_stems:
            if args.dry_run:
                logger.info(f"[Dry Run] Would delete {svg_file}")
            else:
                svg_file.unlink()
                logger.info(f"Deleted {svg_file}")
            delete_count += 1

    if delete_count == 0:
        logger.info("Everything is clean! No missing match files found.")
    else:
        logger.info(f"Total files {'that would be ' if args.dry_run else ''}deleted: {delete_count}")

if __name__ == "__main__":
    main()
