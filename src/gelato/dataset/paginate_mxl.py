# Standard library imports
import re
import copy
import logging
import zipfile
import argparse
import multiprocessing
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Third party imports
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_ids_from_svg(svg_path: Path) -> set:
    """Extracts all injected XML IDs that Verovio rendered onto this specific SVG page."""
    svg_text = svg_path.read_text(encoding="utf-8")
    # Find all id="..." attributes in the SVG
    ids = re.findall(r'id="([^"]+)"', svg_text)
    return set(ids)


def prune_musicxml(xml_root: ET.Element, valid_svg_ids: set) -> ET.Element:
    """
    Deletes measures not present on the SVG page.
    Uses Deep ID Intersection and safely carries over stateful attributes.
    """
    new_root = copy.deepcopy(xml_root)
    
    for part in new_root.findall('.//part'):
        # Keep an actual Element to maintain valid XML structure
        running_attrs = ET.Element('attributes') 
        measures_to_remove = []
        first_measure_kept = False

        for measure in part.findall('measure'):
            # 1. Update the running state with any new attributes
            for attrs in measure.findall('attributes'):
                for child in attrs:
                    # Remove the old attribute of the same type (e.g., old key signature)
                    existing = running_attrs.find(child.tag)
                    if existing is not None:
                        running_attrs.remove(existing)
                    # Append the new updated attribute
                    running_attrs.append(copy.deepcopy(child))

            # 2. Deep ID Intersection: Check EVERY element inside this measure
            measure_ids = {el.get('id') for el in measure.iter() if 'id' in el.attrib}
            
            # If ANY element from this measure made it onto the SVG page, keep the measure
            if measure_ids.intersection(valid_svg_ids):
                if not first_measure_kept:
                    # INJECTION: This is the first measure of the new page.
                    # We must inject the running attributes so xml2abc knows the context!
                    if len(running_attrs) > 0:
                        measure.insert(0, copy.deepcopy(running_attrs))
                    first_measure_kept = True
            else:
                measures_to_remove.append(measure)

        # 3. Safely delete the off-page measures
        for m in measures_to_remove:
            part.remove(m)

    return new_root


def split_mxl_by_svg(mxl_path: Path, svg_dir: Path, output_mxl_dir: Path) -> bool:
    """Finds all SVGs belonging to an MXL, splits the XML, and saves per-page MXLs."""
    try:
        # Find all SVGs generated for this specific MXL stem
        page_svgs = list(svg_dir.glob(f"{mxl_path.stem}_page*.svg"))
        if not page_svgs:
            logger.debug(f"No corresponding SVGs found for {mxl_path.name}")
            return False

        # Extract the raw XML from the MXL archive
        with zipfile.ZipFile(mxl_path, "r") as zf:
            xml_name = next((n for n in zf.namelist() if (n.endswith(".xml") or n.endswith(".musicxml")) and "META" not in n), None)
            if not xml_name:
                logger.debug(f"No valid XML found inside {mxl_path.name}")
                return False
            xml_data = zf.read(xml_name)
            
            # Preserve META-INF if it exists
            meta_data = (
                zf.read("META-INF/container.xml")
                if "META-INF/container.xml" in zf.namelist()
                else None
            )
            
        try:
            original_tree = ET.fromstring(xml_data)
        except Exception as e:
            logger.debug(f"XML parsing failed for {mxl_path.name}: {e}")
            return False

        for svg_path in page_svgs:
            page_suffix = svg_path.stem.split("_")[-1] # Extracts "page1", "page2", etc.
            out_mxl = output_mxl_dir / f"{mxl_path.stem}_{page_suffix}.mxl"
            
            if out_mxl.exists():
                continue

            valid_ids = get_ids_from_svg(svg_path)
            
            # Create the page-specific XML tree
            page_tree = prune_musicxml(original_tree, valid_ids)
            
            # Save the new XML snippet to a compressed MXL archive
            page_xml_data = ET.tostring(page_tree, encoding="unicode")
            with zipfile.ZipFile(out_mxl, "w", zipfile.ZIP_DEFLATED) as zout:
                zout.writestr(xml_name, page_xml_data)
                if meta_data:
                    zout.writestr("META-INF/container.xml", meta_data)

        return True
    except Exception as e:
        logger.debug(f"Failed processing {mxl_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Split MXL files into per-page MXLs based on SVGs."
    )
    parser.add_argument(
        "input_dir", type=str, help="Directory containing .mxl files"
    )
    parser.add_argument(
        "--svg-dir",
        "--svg_dir",
        dest="svg_dir",
        type=str,
        required=True,
        help="Directory containing the rendered .svg pages",
    )
    parser.add_argument(
        "--out-mxl-dir",
        "--out_mxl_dir",
        dest="out_mxl_dir",
        type=str,
        required=True,
        help="Output directory for the per-page MXLs",
    )
    args = parser.parse_args()

    mxl_dir = Path(args.input_dir)
    svg_dir = Path(args.svg_dir)
    out_mxl_dir = Path(args.out_mxl_dir)

    if not mxl_dir.exists() or not mxl_dir.is_dir():
        logger.error(f"Input directory does not exist: {mxl_dir}")
        return
        
    if not svg_dir.exists() or not svg_dir.is_dir():
        logger.error(f"SVG directory does not exist: {svg_dir}")
        return

    out_mxl_dir.mkdir(parents=True, exist_ok=True)

    mxl_files = list(mxl_dir.rglob("*.mxl"))
    if not mxl_files:
        logger.info(f"No .mxl files found in {mxl_dir}")
        return

    logger.info(f"Splitting {len(mxl_files)} MXL files into pages...")

    success_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = {
            executor.submit(split_mxl_by_svg, path, svg_dir, out_mxl_dir): path for path in mxl_files
        }

        with tqdm(total=len(futures), desc="Splitting Pages", unit="file") as pbar:
            for future in as_completed(futures):
                try:
                    if future.result():
                        success_count += 1
                    else:
                        error_count += 1
                except Exception:
                    error_count += 1

                pbar.set_postfix(success=success_count, errors=error_count)
                pbar.update(1)

    logger.info(
        f"Finished processing. Success: {success_count} | Failed: {error_count}"
    )


if __name__ == "__main__":
    main()