import os
import zipfile
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import logging
import argparse

from gelato.data.converter import convert_xml_to_abc
from gelato.data.renderer import Renderer
from gelato.data.canonicalize import canonicalize_abc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_full_dataset(
    input_dir: Path, 
    output_base: Path, 
    split_ratios=(0.8, 0.1, 0.1), 
    limit: int = None,
    seed: int = 42
):
    random.seed(seed)
    
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        return

    # 1. List all MXL files to determine splits
    logger.info("Scanning input directory to build file list...")
    mxl_paths = list(input_dir.rglob("*.mxl"))
    mxl_names = [p.name for p in mxl_paths]
    path_map = {p.name: p for p in mxl_paths}
    
    total_files = len(mxl_names)
    logger.info(f"Total MXL files found: {total_files}")
    
    if total_files == 0:
        return

    if limit:
        mxl_names = random.sample(mxl_names, min(limit, total_files))
        logger.info(f"Limiting to {len(mxl_names)} files as requested.")
        total_files = len(mxl_names)
    else:
        random.shuffle(mxl_names)
        
    # 2. Assign splits
    n_train = int(total_files * split_ratios[0])
    n_val = int(total_files * split_ratios[1])
    
    train_set = set(mxl_names[:n_train])
    val_set = set(mxl_names[n_train:n_train+n_val])
    test_set = set(mxl_names[n_train+n_val:])
    
    logger.info(f"Split sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    # 3. Prepare Directories
    dirs = {
        "train": output_base / "processed_train",
        "validation": output_base / "processed_validation",
        "test": output_base / "processed_test"
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        
    # Temp dirs
    temp_extract_dir = Path("data/temp_full_extract")
    if temp_extract_dir.exists(): shutil.rmtree(temp_extract_dir)
    temp_extract_dir.mkdir(parents=True)
    
    temp_render_dir = Path("data/temp_full_render")
    renderer = Renderer(temp_render_dir)

    # 4. Process files
    logger.info("Starting processing...")
    
    success_count = 0
    error_count = 0
    skipped_count = 0

    for mxl_name in tqdm(mxl_names, desc="Processing"):
        sample_id = Path(mxl_name).stem
        
        # Determine split
        if mxl_name in train_set:
            target_split = "train"
        elif mxl_name in val_set:
            target_split = "validation"
        else:
            target_split = "test"
            
        target_dir = dirs[target_split]
        sample_out = target_dir / sample_id
        
        # Check if already processed
        abc_path = sample_out / "label.abc"
        patches_out = sample_out / "patches"
        
        # Check for atomic completion: label exists AND exactly 4 patches exist
        if sample_out.exists() and abc_path.exists() and patches_out.exists():
            patch_files = list(patches_out.glob("*.png"))
            if len(patch_files) == 4:
                skipped_count += 1
                continue
            
        # Clean up partial, broken, or misconfigured legacy state
        if sample_out.exists():
            shutil.rmtree(sample_out)
            
        sample_out.mkdir(parents=True)
        patches_out.mkdir()
        
        try:
            mxl_source = path_map[mxl_name]
            
            # Unzip XML
            xml_path = temp_extract_dir / f"{sample_id}.xml"
            valid_xml = False
            with zipfile.ZipFile(mxl_source, 'r') as z:
                xml_files = [f for f in z.namelist() if f.endswith(".xml") and not f.startswith("META-INF")]
                if xml_files:
                    with open(xml_path, "wb") as f_out:
                        f_out.write(z.read(xml_files[0]))
                    valid_xml = True
            
            error = None
            if valid_xml:
                # Convert
                convert_xml_to_abc(xml_path, abc_path)
                canonicalize_abc(abc_path)
                
                # Render
                svgs = renderer.render_abc_to_svg(abc_path)
                
                if svgs:
                    png = renderer.convert_svg_to_png(svgs[0])
                    segments = renderer.process_image_for_model(png)
                    
                    p_idx = 0
                    for seg in segments: # Usually exactly 1 segment due to renderer updates
                        patches = renderer.slice_segment(seg) # Usually exactly 4 patches
                        for p in patches:
                            p.save(patches_out / f"{p_idx}.png")
                            p_idx += 1
                    
                    # Verify generation completed correctly
                    patch_list = list(patches_out.glob("*.png"))
                    if len(patch_list) != 4:
                        error = f"Generated {len(patch_list)} patches instead of exactly 4."
                    else:
                        success_count += 1
                else:
                    error = "No SVGs rendered."
            else:
                error = "Invalid MXL or no XML found inside."
            
            if error:
                logger.warning(f"Error for {sample_id}: {error}")
                error_count += 1
                if sample_out.exists():
                    shutil.rmtree(sample_out) # Clean up partial state on failure
            
            # Cleanup single XML
            if xml_path.exists(): xml_path.unlink()
            
        except Exception as e:
            logger.error(f"Exception processing {mxl_name}: {e}")
            error_count += 1
            if sample_out.exists():
                shutil.rmtree(sample_out) # Clean up partial state on failure
                
    # Cleanup dirs
    logger.info("Processing loop finished. Cleaning up temporary directories...")
    if temp_extract_dir.exists(): shutil.rmtree(temp_extract_dir)
    if temp_render_dir.exists(): shutil.rmtree(temp_render_dir)
    
    logger.info(f"Done. Success: {success_count}, Skipped: {skipped_count}, Errors: {error_count}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/mxl", help="Input directory containing .mxl files")
    parser.add_argument("--out", default="data")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process (for testing)")
    args = parser.parse_args()
    
    create_full_dataset(Path(args.input), Path(args.out), limit=args.limit)

if __name__ == "__main__":
    main()
