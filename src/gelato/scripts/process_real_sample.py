import os
import zipfile
from pathlib import Path
import shutil
from gelato.data.converter import convert_xml_to_abc
from gelato.data.renderer import Renderer
from gelato.data.canonicalize import canonicalize_abc

def main():
    # Input/Output dirs
    mxl_dir = Path("data/mxl_sample")
    processed_dir = Path("data/processed_real")
    
    # Clean output
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True)
    
    mxl_files = list(mxl_dir.glob("*.mxl"))
    print(f"Found {len(mxl_files)} MXL files.")
    
    renderer = Renderer(Path("data/render_real"))

    for mxl_path in mxl_files:
        sample_id = mxl_path.stem
        print(f"Processing {sample_id}...")
        
        # 1. Unzip MXL to get XML
        # MXL is a zip. It usually contains META-INF/container.xml and the score.xml
        try:
            with zipfile.ZipFile(mxl_path, 'r') as z:
                # Find the main xml file. Usually ending in .xml
                xml_files = [f for f in z.namelist() if f.endswith(".xml") and not f.startswith("META-INF")]
                if not xml_files:
                    print(f"Skipping {sample_id}: No XML found inside MXL.")
                    continue
                
                # Extract to temp
                # We rename it to sample_id.xml
                xml_content = z.read(xml_files[0])
                xml_path = mxl_dir / f"{sample_id}.xml"
                with open(xml_path, "wb") as f:
                    f.write(xml_content)
                    
        except Exception as e:
            print(f"Failed to unzip {mxl_path}: {e}")
            continue

        try:
            # 2. Convert to ABC
            abc_path = processed_dir / sample_id / "label.abc" # Temp location? No pipeline expects structure.
            # Convert needs output path
            # converter expects output path including filename
            
            # Prepare sample folder
            sample_folder = processed_dir / sample_id
            sample_folder.mkdir(parents=True, exist_ok=True)
            patches_dir = sample_folder / "patches"
            patches_dir.mkdir(exist_ok=True)
            
            generated_abc = convert_xml_to_abc(xml_path, sample_folder / "label.abc")
            
            # 3. Canonicalize
            canonicalize_abc(generated_abc)
            
            # 4. Render
            svgs = renderer.render_abc_to_svg(generated_abc)
            if not svgs:
                print(f"No SVGs rendered for {sample_id}")
                continue
                
            # 5. Slice
            # Take first page/svg for now
            png = renderer.convert_svg_to_png(svgs[0])
            segments = renderer.process_image_for_model(png)
            
            # Setup patches
            patch_idx = 0
            for segment in segments:
                patches = renderer.slice_segment(segment)
                for patch in patches:
                    patch.save(patches_dir / f"{patch_idx}.png")
                    patch_idx += 1
            
            print(f"Finished {sample_id}: {patch_idx} patches.")
            
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
        finally:
            # Cleanup temp xml
            if xml_path.exists():
                xml_path.unlink()

if __name__ == "__main__":
    main()
