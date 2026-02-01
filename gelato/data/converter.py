import subprocess
import logging
from pathlib import Path
import os
import sys

logger = logging.getLogger(__name__)

def convert_xml_to_abc(xml_path: Path, output_path: Path) -> Path:
    """
    Converts a MusicXML file to ABC format using the bundled xml2abc.py script.
    
    Args:
        xml_path: Path to the input MusicXML file.
        output_path: Path where the output ABC file should be saved.
        
    Returns:
        Path to the generated ABC file.
    """
    script_path = Path(__file__).parent / "xml2abc.py"
    
    if not script_path.exists():
        raise FileNotFoundError(f"xml2abc.py not found at {script_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # xml2abc.py usage: python3 xml2abc.py [options] <file>
    # -o <dir> : output directory
    # By default it writes to the same directory or stdout. 
    # We will use -o to specify the output directory, but xml2abc infers filename from input.
    # So we might need to rename it afterwards if output_path has a specific name.
    
    cmd = [
        sys.executable,
        str(script_path),
        str(xml_path),
        "-o", str(output_path.parent), # Output directory
        "-d", "8", # Force unit length to 1/8
        "-b", "5", # Force 5 bars per line
    ]
    
    logger.info(f"Running conversion: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"xml2abc failed: {e.stderr}")
        raise RuntimeError(f"Failed to convert {xml_path}: {e.stderr}") from e
        
    # xml2abc usually saves as <filename>.abc in the output folder.
    # Let's find the expected output file.
    # It replaces extensions like .xml, .mxl, .musicxml with .abc
    
    possible_name = xml_path.stem + ".abc"
    generated_file = output_path.parent / possible_name
    
    if not generated_file.exists():
        # Sometimes it might just append .abc?
        # Let's list files in the directory to be sure or check logic.
        # But generally it replaces extension.
        logger.warning(f"Expected generated file {generated_file} not found. Checking if it was named differently.")
        # Fallback check
        if (output_path.parent / (xml_path.name + ".abc")).exists():
             generated_file = output_path.parent / (xml_path.name + ".abc")
        else:
             raise FileNotFoundError(f"Could not locate the generated ABC file for {xml_path}")

    # If the user requested a specific output path (filename), rename it
    if generated_file != output_path:
        generated_file.rename(output_path)
        
    return output_path
