"""
strip_abc_sync.py — Synchronized ABC stripper aligned with the tokenizer vocabulary.

Changes from original strip_abc.py:
  - V: lines: strips nm=, snm=, AND simplifies clef values to match tokenizer
  - Strips voice attributes not in the tokenizer vocab (merge, stem=, dyn=, etc.)
  - Strips stafflines=, staffscale=, middle= (not in tokenizer)
  - Preserves clef= since tokenizer_sync.py now has "clef=" token
  - Strips %abc header line
  - Strips text annotations ("^above", "_below", "<left", ">right", "@x,y offset")
    while preserving chord symbols ("Am7", "Dm", "F#7")
"""

import logging
import argparse
import multiprocessing
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Regexes ──────────────────────────────────────────────────────────────

# Headers to KEEP (music-essential)
_KEEP_HEADERS = re.compile(r"^[MLKVQP]:")
# Any standard header line
_ANY_HEADER = re.compile(r"^[a-zA-Z]:")
# Directives (%%...)
_DROP_DIRECTIVE = re.compile(r"^%%")
# ABC version header
_ABC_HEADER = re.compile(r"^%abc")
# Inline comments (% followed by anything, at end of line)
_INLINE_COMMENT = re.compile(r"\s*%.*$")
# Voice display names (not useful for music generation)
_VOICE_NAMES = re.compile(r'\s+(?:nm|snm|name|sname)="[^"]*"')
# Voice attributes NOT in the tokenizer vocab
_VOICE_STRIP_ATTRS = re.compile(
    r'\s+(?:merge|up|down|stem|gstem|dyn|lyrics|middle|staffscale|stafflines)'
    r'(?:=[^\s]*)?'
)
# Text annotations: "^above" "_below" "<left" ">right" "@x,y offset"
# Chord symbols start with A-G and are KEPT: "Am7" "Dm" "F#7"
_TEXT_ANNOTATION = re.compile(r'"[\^_<>@][^"]*"')


def strip_abc(text: str) -> str:
    """Return a minimal ABC string suitable for tokenization and training.

    Keeps: M:, L:, K:, V:, Q:, P: headers, music lines, and chord symbols.
    Strips: all other headers, %% directives, inline comments,
            voice display names, non-musical voice attributes,
            and text annotations ("^...", "_...", "<...", ">...", "@...").
    """
    out = []
    for line in text.splitlines():
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Skip %abc version line
        if _ABC_HEADER.match(stripped):
            continue

        # Check if it's a header line
        if _ANY_HEADER.match(stripped):
            if not _KEEP_HEADERS.match(stripped):
                continue
        # Drop directives
        elif _DROP_DIRECTIVE.match(stripped):
            continue

        # Clean V: lines
        if stripped.startswith("V:"):
            line = _VOICE_NAMES.sub("", line)
            line = _VOICE_STRIP_ATTRS.sub("", line)

        # Remove inline comments
        line = _INLINE_COMMENT.sub("", line).rstrip()

        # Remove text annotations ("^above", "_below", "<left", ">right", "@x,y...")
        # but keep chord symbols ("Am7", "Dm", "F#7" — start with A-G)
        line = _TEXT_ANNOTATION.sub("", line).strip()

        if line.strip():
            out.append(line.strip())

    return "\n".join(out) + "\n"


def process_file(abc_path: Path, output_dir: Path) -> bool:
    """Strip one ABC file and write to output_dir."""
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
        description="Strip non-essential headers and comments from ABC files."
    )
    parser.add_argument("input_dir", type=str, help="Directory containing .abc files")
    parser.add_argument(
        "--out-dir", "--out_dir", dest="out_dir",
        type=str, required=True,
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

    logger.info(f"Finished. Success: {success_count} | Failed: {error_count}")


if __name__ == "__main__":
    main()