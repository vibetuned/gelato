from pathlib import Path
import re

def canonicalize_abc(abc_path: Path, output_path: Path = None):
    """
    Cleans and standardizes an ABC file according to Gelato rules:
    - Replace Title/Composer text with <text>
    - Remove lyrics (w: lines)
    - Insert $ at the end of every music line
    - Ensure L:1/8 is preserved (already handled by converter, but check?)
    """
    if output_path is None:
        output_path = abc_path

    with open(abc_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    clean_lines = []
    
    # Regex for standard ABC headers: Letter followed by colon
    header_pattern = re.compile(r"^([A-Za-z]):(.*)")
    
    # Fields to anonymize
    text_fields = {'T', 'C', 'A', 'N', 'Z', 'O', 'R', 'H', 'B', 'D', 'F', 'S'} 
    # T=Title, C=Composer, A=Area, N=Notes, Z=Transcription, etc.
    
    # Fields to strip completely (Lyrics)
    strip_fields = {'w', 'W'}

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = header_pattern.match(line)
        if match:
            key, value = match.groups()
            
            if key in strip_fields:
                continue # Skip lyrics
            
            if key in text_fields:
                # Replace content with <text>
                clean_lines.append(f"{key}: <text>")
            else:
                # Keep other headers (M, K, L, V, Q, P) as is
                clean_lines.append(line)
        elif line.startswith('%'):
             # Comment
             continue
        else:
            # Assume it's a music line
            # Check if it looks like music (basic heuristic)
            # If it contains notes, bars, etc.
            # Append $
            # If line ends with backslash (continuation), put $ before or after?
            # Standard ABC: \ means "continue line".
            # If we want to mark "visual line break" for the model, we want $ where the image breaks.
            # xml2abc -b 5 creates new lines in the file.
            # So every line in the file is a visual line (system).
            # We append $ to indicate "End of System".
            
            # Handle continuation char
            if line.endswith('\\'):
                # remove \ , add $, maybe re-add \ if strict ABC requires it?
                # But usually newlines in ABC are just spaces unless it defaults to something else.
                # Actually, in standard ABC, a line break is a line break.
                # \ is used to join lines.
                # If xml2abc output uses \, it means "this is logically one line".
                # But visually it might be split?
                # With -b 5, xml2abc breaks lines after 5 bars. It does NOT use \ usually.
                pass
                
            clean_lines.append(line + " $")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(clean_lines) + '\n')
    
    return output_path
