# Standard library imports
import logging
import argparse
import multiprocessing
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Third party imports
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Header fields that are strictly necessary for music representation
_KEEP_HEADERS = re.compile(r"^[MLKVQP]:")
# Matches any standard abc header line (single letter then :)
_ANY_HEADER = re.compile(r"^[a-zA-Z]:")
# Lines starting with %% (abc directives / pseudo-comments)
_DROP_DIRECTIVE = re.compile(r"^%%")
# Inline bar-count comments at end of note lines (e.g. " %7")
_INLINE_COMMENT = re.compile(r"\s*%.*$")
# Voice name attributes like nm="Piano" snm="Pno."
_VOICE_NAMES = re.compile(r'\s+(?:nm|snm)="[^"]*"')


def strip_abc(text: str) -> str:
    """Return a minimal ABC string with metadata and comments removed.
    Preserves inline annotations and decorations (like "tr", !p!)."""
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        
        # Check if it's a header line
        if _ANY_HEADER.match(stripped):
            if not _KEEP_HEADERS.match(stripped):
                continue
        # Drop directives
        elif _DROP_DIRECTIVE.match(stripped):
            continue
            
        # Remove voice name attributes from V: lines
        if stripped.startswith("V:"):
            line = _VOICE_NAMES.sub("", line)
            
        # Remove inline comments from all other lines
        line = _INLINE_COMMENT.sub("", line).rstrip()
        
        if line.strip():  # Don't add completely empty lines
            out.append(line.strip())
            
    return "\n".join(out) + "\n"


def process_file(abc_path: Path, output_dir: Path) -> bool:
    """Strip metadata/comments from a single ABC file and write to output_dir."""
    try:
        out_path = output_dir / abc_path.name
        if out_path.exists():
            return True
        text = abc_path.read_text(encoding="utf-8")
        out_path.write_text(strip_abc(text), encoding="utf-8")
        return True
    except Exception as e:
        logger.debug(f"Failed processing {abc_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Strip X:, T:, C:, I: headers and comments from ABC files."
    )
    parser.add_argument("input_dir", type=str, help="Directory containing .abc files")
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=str,
        required=True,
        help="Output directory for stripped .abc files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    abc_files = list(input_dir.rglob("*.abc"))
    if not abc_files:
        logger.info(f"No .abc files found in {input_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Stripping {len(abc_files)} ABC files...")

    success_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = {
            executor.submit(process_file, path, out_dir): path for path in abc_files
        }
        with tqdm(total=len(futures), desc="Stripping ABC", unit="file") as pbar:
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
        f"Finished. Success: {success_count} | Failed: {error_count}"
    )


if __name__ == "__main__":
    main()
