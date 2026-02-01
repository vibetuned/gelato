import os
import tarfile
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
    tar_path: Path, 
    output_base: Path, 
    split_ratios=(0.8, 0.1, 0.1), 
    limit: int = None,
    seed: int = 42
):
    random.seed(seed)
    
    if not tar_path.exists():
        logger.error(f"Archive not found: {tar_path}")
        return

    # 1. First Pass: List all MXL files to determine splits
    logger.info("Scanning archive to build file list...")
    mxl_names = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tqdm(tar, desc="Scanning"):
            if member.isfile() and member.name.endswith(".mxl"):
                mxl_names.append(member.name)
    
    total_files = len(mxl_names)
    logger.info(f"Total MXL files found: {total_files}")
    
    if limit:
        mxl_names = random.sample(mxl_names, min(limit, total_files))
        logger.info(f"Limiting to {len(mxl_names)} files as requested.")
        total_files = len(mxl_names)
    else:
        random.shuffle(mxl_names)
        
    # 2. Assign splits
    n_train = int(total_files * split_ratios[0])
    n_val = int(total_files * split_ratios[1])
    # Rest to test
    
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

    # 4. Second Pass: Process
    # We iterate again. This is slower than random access if tar is small, 
    # but for large compressed tar, sequential read is better than seeking?
    # Actually, iterate and check if name is in our selected sets.
    
    # Optimization: If we just iterate tar again, we don't need random seeking.
    
    logger.info("Starting processing...")
    
    success_count = 0
    error_count = 0
    
    skipped_count = 0
    found_count = 0

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tqdm(tar, desc="Processing", total=len(mxl_names) if limit else None): # Total is approximate if not limited
            if member.name not in train_set and member.name not in val_set and member.name not in test_set:
                continue
                
            sample_id = Path(member.name).stem
            
            # Determine split
            if member.name in train_set:
                target_split = "train"
            elif member.name in val_set:
                target_split = "validation"
            else:
                target_split = "test"
                
            target_dir = dirs[target_split]
            sample_out = target_dir / sample_id
            
            # We found a relevant file
            found_count += 1
            
            # Check if already processed
            # We check for label.abc as a marker of completion (or at least partial success)
            if (sample_out / "label.abc").exists():
                # logger.info(f"Skipping {sample_id} (already exists)")
                skipped_count += 1
                if found_count >= total_files:
                    logger.info("Processed all target files (some skipped). Exiting loop early.")
                    break
                continue
            
            try:
                # Extract
                tar.extract(member, path=temp_extract_dir, filter='data')
                mxl_path = temp_extract_dir / member.name
                
                # Unzip XML
                xml_path = temp_extract_dir / f"{sample_id}.xml"
                valid_xml = False
                with zipfile.ZipFile(mxl_path, 'r') as z:
                    xml_files = [f for f in z.namelist() if f.endswith(".xml") and not f.startswith("META-INF")]
                    if xml_files:
                        with open(xml_path, "wb") as f_out:
                            f_out.write(z.read(xml_files[0]))
                        valid_xml = True
                
                # Setup output folder
                if sample_out.exists(): shutil.rmtree(sample_out)
                sample_out.mkdir(parents=True)
                patches_out = sample_out / "patches"
                patches_out.mkdir()
                
                if valid_xml:
                    # Convert
                    abc_path = sample_out / "label.abc"
                    convert_xml_to_abc(xml_path, abc_path)
                    canonicalize_abc(abc_path)
                    
                    # Render
                    svgs = renderer.render_abc_to_svg(abc_path)
                    
                    if svgs:
                        # Slice (First Page for now to keep size manageable? Or all pages?)
                        # Paper says "Slicing Logic" - implies full score.
                        # But typically datasets split pages or lines.
                        # Our current logic handles single image -> patches.
                        # If multiple SVGs (pages), we should process all?
                        # If we process all, we have multiple 'samples' or one long sample?
                        # GelatoDataset expects one label file.
                        # If we have multiple pages, ABC is usually one file.
                        # So we might need to concatenate images or treat them as sequence?
                        # For now, let's Stick to Page 1 to ensure 1 sample = 1 ABC file mapping.
                        # Multi-page OMR is complex.
                        
                        png = renderer.convert_svg_to_png(svgs[0])
                        segments = renderer.process_image_for_model(png)
                        
                        p_idx = 0
                        for seg in segments:
                            patches = renderer.slice_segment(seg)
                            for p in patches:
                                p.save(patches_out / f"{p_idx}.png")
                                p_idx += 1
                        success_count += 1
                    else:
                        logger.warning(f"No SVGs for {sample_id}")
                        error_count += 1
                
                # Cleanup single file (mxl and xml)
                if mxl_path.exists(): mxl_path.unlink()
                if xml_path.exists(): xml_path.unlink()
                
            except Exception as e:
                logger.error(f"Error processing {member.name}: {e}")
                error_count += 1
            
            if found_count >= total_files:
                logger.info("Processed all target files. Exiting loop early.")
                break
                
    # Cleanup dirs
    logger.info("Processing loop finished. Cleaning up temporary directories...")
    shutil.rmtree(temp_extract_dir)
    shutil.rmtree(temp_render_dir)
    
    logger.info(f"Done. Success: {success_count}, Skipped: {skipped_count}, Errors: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", default="data/mxl.tar.gz")
    parser.add_argument("--out", default="data")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process (for testing)")
    args = parser.parse_args()
    
    create_full_dataset(Path(args.tar), Path(args.out), limit=args.limit)
