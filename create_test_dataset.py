import os
import tarfile
import zipfile
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import logging

from gelato.data.converter import convert_xml_to_abc
from gelato.data.renderer import Renderer
from gelato.data.canonicalize import canonicalize_abc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dataset(
    tar_path: Path, 
    output_dir: Path, 
    num_samples: int = 20, 
    seed: int = 42
):
    """
    Extracts `num_samples` mxl files from `tar_path`, processes them, 
    and saves them to `output_dir`.
    """
    random.seed(seed)
    
    if not tar_path.exists():
        logger.error(f"Archive not found: {tar_path}")
        return

    # 1. List valid files in tar
    logger.info("Reading tar archive listing... (this might take a moment)")
    mxl_members = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".mxl"):
                mxl_members.append(member)
    
    logger.info(f"Found {len(mxl_members)} MXL files in archive.")
    
    if len(mxl_members) == 0:
        return
        
    # Sample
    if len(mxl_members) > num_samples:
        members_to_extract = random.sample(mxl_members, num_samples)
    else:
        members_to_extract = mxl_members
        
    logger.info(f"Extracting {len(members_to_extract)} samples...")
    
    # Prep directories
    temp_extract_dir = Path("data/temp_mxl_extract")
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)
    temp_extract_dir.mkdir(parents=True)
    
    if output_dir.exists():
        logger.warning(f"Output directory {output_dir} exists. Skipping cleanup to avoid accidental data loss.")
        # shutil.rmtree(output_dir)
    else:
        output_dir.mkdir(parents=True)
    
    renderer = Renderer(Path("data/temp_render"))

    # Extract and Process
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tqdm(members_to_extract, desc="Processing"):
            try:
                # Extract MXL
                tar.extract(member, path=temp_extract_dir, filter='data')
                mxl_path = temp_extract_dir / member.name
                
                sample_id = Path(member.name).stem
                
                # Unzip MXL -> XML
                xml_path = temp_extract_dir / f"{sample_id}.xml"
                valid_xml = False
                
                with zipfile.ZipFile(mxl_path, 'r') as z:
                    xml_files = [f for f in z.namelist() if f.endswith(".xml") and not f.startswith("META-INF")]
                    if xml_files:
                        with open(xml_path, "wb") as f_out:
                            f_out.write(z.read(xml_files[0]))
                        valid_xml = True
                
                if not valid_xml:
                    continue
                    
                # Convert to ABC
                sample_out_dir = output_dir / sample_id
                sample_out_dir.mkdir(parents=True, exist_ok=True)
                patches_dir = sample_out_dir / "patches"
                patches_dir.mkdir(exist_ok=True)
                
                abc_path = sample_out_dir / "label.abc"
                convert_xml_to_abc(xml_path, abc_path)
                canonicalize_abc(abc_path)
                
                # Render
                svgs = renderer.render_abc_to_svg(abc_path)
                if not svgs:
                    continue
                    
                # Slice (First Page Only for Speed)
                png = renderer.convert_svg_to_png(svgs[0])
                segments = renderer.process_image_for_model(png)
                
                patch_cnt = 0
                for segment in segments:
                    patches = renderer.slice_segment(segment)
                    for patch in patches:
                         patch.save(patches_dir / f"{patch_cnt}.png")
                         patch_cnt += 1
                         
            except Exception as e:
                logger.error(f"Failed sample {member.name}: {e}")
                
    # Cleanup
    shutil.rmtree(temp_extract_dir)
    shutil.rmtree("data/temp_render") # Clean render temp
    logger.info(f"Created dataset at {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", default="data/mxl.tar.gz")
    parser.add_argument("--out", default="data/dataset_small")
    parser.add_argument("--num", type=int, default=20)
    args = parser.parse_args()
    
    create_dataset(Path(args.tar), Path(args.out), args.num)
