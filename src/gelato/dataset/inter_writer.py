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

# Medium Vocabulary
KEYS = ["C", "G", "F", "D", "Bb", "A", "Eb"]
TIME_SIGNATURES = ["4/4", "3/4"]

# Safe Rhythm Templates (Guaranteed to perfectly fill the measure)
RHYTHMS_4_4 = [
    [1.0, 1.0, 1.0, 1.0],         # Four quarters
    [2.0, 2.0],                   # Two halfs
    [0.5, 0.5, 1.0, 2.0],         # Two eighths, quarter, half
    [1.0, 0.5, 0.5, 2.0],         # Quarter, two eighths, half
    [0.5, 0.5, 0.5, 0.5, 2.0],    # Four eighths, half
    [1.0, 2.0, 1.0],              # Syncopated quarter, half, quarter
    [1.5, 0.5, 2.0],              # Dotted quarter, eighth, half
    [3.0, 1.0],                   # Dotted half, quarter
]

RHYTHMS_3_4 = [
    [1.0, 1.0, 1.0],              # Three quarters
    [2.0, 1.0],                   # Half, quarter
    [1.0, 2.0],                   # Quarter, half
    [0.5, 0.5, 1.0, 1.0],         # Two eighths, two quarters
]

def generate_medium_score() -> music21.stream.Score:
    """Creates a 4 to 8 measure, single-staff continuous melody."""
    score = music21.stream.Score()
    part = music21.stream.Part()
    
    ts_str = random.choice(TIME_SIGNATURES)
    ts = music21.meter.TimeSignature(ts_str)
    key_obj = music21.key.Key(random.choice(KEYS))
    clef_obj = random.choice([music21.clef.TrebleClef(), music21.clef.BassClef()])
    
    num_measures = random.randint(4, 8)
    
    # Start the melody on a stable degree (Tonic, Mediant, or Dominant)
    current_degree = random.choice([1, 3, 5]) 
    
    for m_idx in range(1, num_measures + 1):
        measure = music21.stream.Measure(number=m_idx)
        
        # Inject stateful attributes only on the first measure
        if m_idx == 1:
            measure.timeSignature = ts
            measure.keySignature = music21.key.KeySignature(key_obj.sharps)
            measure.clef = clef_obj
            
        # The Final Measure: Force a long note on the Tonic to close the phrase
        if m_idx == num_measures:
            dur = ts.barDuration.quarterLength
            p = key_obj.pitchFromDegree(1)
            if isinstance(clef_obj, music21.clef.BassClef):
                p.octave -= 1
            measure.insert(0.0, music21.note.Note(p, quarterLength=dur))
            part.append(measure)
            break

        # Standard Measures: Pick a random rhythm template
        rhythm_pool = RHYTHMS_4_4 if ts_str == "4/4" else RHYTHMS_3_4
        rhythm = random.choice(rhythm_pool)
        
        offset = 0.0
        for dur in rhythm:
            if random.random() < 0.10: # 10% chance to drop a rest
                measure.insert(offset, music21.note.Rest(quarterLength=dur))
            else:
                # Diatonic Random Walk: Move up/down by 1 or 2 scale steps
                step = random.choice([-2, -1, 1, 2])
                
                # Keep the melody strictly contained within an octave and a half
                current_degree = max(1, min(10, current_degree + step))
                
                p = key_obj.pitchFromDegree(current_degree)
                if isinstance(clef_obj, music21.clef.BassClef):
                    p.octave -= 1
                
                n = music21.note.Note(p, quarterLength=dur)
                measure.insert(offset, n)
            
            offset += dur
            
        part.append(measure)
        
    score.insert(0, part)
    return score

def process_file(args_tuple) -> bool:
    file_idx, output_dir = args_tuple
    try:
        out_path = output_dir / f"medium_bridge_{file_idx:06d}.mxl"
        if out_path.exists():
            return True
            
        score = generate_medium_score()
        score.write('musicxml', fp=out_path)
        return True
    except Exception as e:
        logger.error(f"Failed to generate score {file_idx}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate medium-complexity scores for Curriculum Learning.")
    parser.add_argument("output_dir", type=str, help="Output directory for medium MXL files")
    parser.add_argument("num_files", type=int, help="Number of files to generate (e.g., 10000)")
    
    args = parser.parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating {args.num_files} medium files in {output_path}...")
    
    tasks = [(i, output_path) for i in range(args.num_files)]
    success_count, error_count = 0, 0

    with multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() - 1)) as pool:
        results = pool.imap_unordered(process_file, tasks)
        with tqdm(total=args.num_files, desc="Generating Medium Scores") as pbar:
            for is_success in results:
                if is_success:
                    success_count += 1
                else:
                    error_count += 1
                pbar.update(1)

    logger.info(f"Finished. Generated {success_count} files (Failed: {error_count}).")

if __name__ == "__main__":
    main()