# Standard library imports
import logging
import zipfile
import argparse
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Third party imports
from tqdm import tqdm

from gelato.data.xml2abc import vertaal

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def convert_mxl_to_abc(mxl_path: Path, output_dir: Path) -> bool:
    """Convert a MusicXML (.mxl) file to ABC format."""
    try:
        with zipfile.ZipFile(mxl_path, "r") as zf:
            xml_name = next(
                (
                    name
                    for name in zf.namelist()
                    if (name.endswith(".xml") or name.endswith(".musicxml"))
                    and name != "META-INF/container.xml"
                ),
                None,
            )
            if not xml_name:
                return False
            data = zf.read(xml_name).decode("utf-8")

        # Call xml2abc.vertaal which returns a tuple (abc_string, info_string)
        abc_text, _ = vertaal(data)

        if not abc_text:
            return False

        out_abc = output_dir / f"{mxl_path.stem}.abc"
        out_abc.write_text(abc_text, encoding="utf-8")

        return True

    except zipfile.BadZipFile:
        pass  # Silently fail on bad zips to keep console clean
    except Exception as e:
        logger.debug(f"Error processing {mxl_path.name}: {e}")

    return False


def _worker(args):
    mxl_path, output_dir = args
    return mxl_path, convert_mxl_to_abc(mxl_path, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MXL files to ABC files."
    )
    parser.add_argument("input_dir", type=str, help="Directory containing .mxl files")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/output_abc",
        help="Output directory for ABC files",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.exists() or not input_path.is_dir():
        logger.error(f"Input directory does not exist: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    mxl_files = list(input_path.rglob("*.mxl"))
    if not mxl_files:
        logger.info(f"No .mxl files found in {input_path}")
        return

    tasks = [(f, output_path) for f in mxl_files]
    success_count = 0
    error_count = 0

    workers = max(1, multiprocessing.cpu_count() - 2)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_worker, task) for task in tasks]

        with tqdm(total=len(mxl_files), desc="Converting MXL to ABC", unit="file") as pbar:
            for future in as_completed(futures):
                try:
                    _, is_success = future.result()
                except Exception:
                    is_success = False

                if is_success:
                    success_count += 1
                else:
                    error_count += 1

                pbar.set_postfix(success=success_count, errors=error_count)
                pbar.update(1)

    logger.info(f"Finished conversion. Success: {success_count} | Failed: {error_count}")


if __name__ == "__main__":
    main()
