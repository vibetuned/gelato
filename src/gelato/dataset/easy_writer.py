import os
import random
import logging
import argparse
import multiprocessing
from pathlib import Path
from tqdm import tqdm

try:
    import music21
    from music21 import environment
    environment.UserSettings()["warnings"] = 0
except ImportError:
    pass

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# Basic vocabulary for Phase 1 Alignment
KEYS = ["C", "G", "F", "D", "Bb"]
TIME_SIGNATURES = ["4/4", "3/4", "2/4"]
DURATIONS = [4.0, 2.0, 1.0, 0.5]

def generate_single_note(measure: music21.stream.Measure, bar_length: float, key_obj: music21.key.Key):
    """Generates a single diatonic note or rest."""
    dur = random.choice([d for d in DURATIONS if d <= bar_length])
    
    if random.random() < 0.2:
        n = music21.note.Rest(quarterLength=dur)
    else:
        # Pick a random degree from the current key's scale (1 through 7)
        degree = random.randint(1, 7)
        pitch = key_obj.pitchFromDegree(degree)
        
        # Assign an easy-to-read octave
        pitch.octave = random.choice([4, 5]) if isinstance(measure.clef, music21.clef.TrebleClef) else random.choice([2, 3])
        n = music21.note.Note(pitch, quarterLength=dur)
            
    measure.insert(0.0, n)
    
    # Pad remainder of the measure with a rest
    if bar_length > dur:
        measure.insert(dur, music21.note.Rest(quarterLength=bar_length - dur))

def generate_basic_chord(measure: music21.stream.Measure, bar_length: float, key_obj: music21.key.Key):
    """Generates a strictly diatonic chord to avoid bad accidentals."""
    dur = random.choice([d for d in DURATIONS if d <= bar_length])
    
    # Pick a standard diatonic chord for the key
    rn_str = random.choice(["I", "ii", "IV", "V", "vi"])
    rn = music21.roman.RomanNumeral(rn_str, key_obj)
    
    # Shift the chord to the correct staff
    octave_shift = 0 if isinstance(measure.clef, music21.clef.TrebleClef) else -1
    pitches = [p.transpose(octave_shift * 12) for p in rn.pitches]
    
    c = music21.chord.Chord(pitches, quarterLength=dur)
    measure.insert(0.0, c)

    # Pad remainder of the measure with a rest
    if bar_length > dur:
        measure.insert(dur, music21.note.Rest(quarterLength=bar_length - dur))

def generate_simple_melody(measure: music21.stream.Measure, bar_length: float, key_obj: music21.key.Key):
    """Generates 2 to 4 consecutive notes by walking the diatonic scale."""
    num_notes = random.randint(2, 4)
    dur = bar_length / num_notes
    
    start_degree = random.randint(1, 5)
    direction = 1 if random.random() < 0.5 else -1 # 1 for Ascending, -1 for Descending
    
    offset = 0.0
    for i in range(num_notes):
        # Calculate the next scale degree, wrapping around the octave safely
        current_degree = start_degree + (i * direction)
        
        # Get the strict diatonic pitch from the key
        p = key_obj.pitchFromDegree(current_degree)
        
        # Lock to a safe octave
        p.octave = 4 if isinstance(measure.clef, music21.clef.TrebleClef) else 3
        
        n = music21.note.Note(p, quarterLength=dur)
        measure.insert(offset, n)
        offset += dur

def generate_easy_score() -> music21.stream.Score:
    """Creates a strictly 1-measure, single-staff score."""
    score = music21.stream.Score()
    part = music21.stream.Part()
    
    ts = music21.meter.TimeSignature(random.choice(TIME_SIGNATURES))
    key_obj = music21.key.Key(random.choice(KEYS))
    
    measure = music21.stream.Measure(number=1)
    measure.timeSignature = ts
    measure.keySignature = music21.key.KeySignature(key_obj.sharps)
    measure.clef = random.choice([music21.clef.TrebleClef(), music21.clef.BassClef()])
    
    mode = random.choice(["note", "chord", "melody"])
    
    # Pass the key_obj down so the generators know what scale to use
    if mode == "note":
        generate_single_note(measure, ts.barDuration.quarterLength, key_obj)
    elif mode == "chord":
        generate_basic_chord(measure, ts.barDuration.quarterLength, key_obj)
    else:
        generate_simple_melody(measure, ts.barDuration.quarterLength, key_obj)
        
    part.append(measure)
    score.insert(0, part)
    return score

# ... (keep process_file and main exactly the same as the previous script)

def process_file(args_tuple) -> bool:
    file_idx, output_dir = args_tuple
    try:
        out_path = output_dir / f"easy_align_{file_idx:06d}.mxl"
        if out_path.exists():
            return True
            
        score = generate_easy_score()
        # No complex tie-stripping or makeNotation passes needed for 1 measure
        score.write('musicxml', fp=out_path)
        return True
    except Exception as e:
        logger.error(f"Failed to generate score {file_idx}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate micro-scores for VLM Modality Alignment.")
    parser.add_argument("output_dir", type=str, help="Output directory for easy MXL files")
    parser.add_argument("num_files", type=int, help="Number of files to generate (e.g., 5000)")
    
    args = parser.parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating {args.num_files} easy files in {output_path}...")
    
    tasks = [(i, output_path) for i in range(args.num_files)]
    success_count, error_count = 0, 0

    with multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() - 1)) as pool:
        results = pool.imap_unordered(process_file, tasks)
        with tqdm(total=args.num_files, desc="Generating Micro-Scores") as pbar:
            for is_success in results:
                if is_success:
                    success_count += 1
                else:
                    error_count += 1
                pbar.update(1)

    logger.info(f"Finished. Generated {success_count} files (Failed: {error_count}).")

if __name__ == "__main__":
    main()