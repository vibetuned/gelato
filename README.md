# Gelato

Gelato is an Optical Music Recognition (OMR) system using a Gemma-based vision-language model.

## Setup

### System Dependencies

Execute the following command to install required system packages on Debian/Ubuntu:

```bash
sudo apt-get update && sudo apt-get install -y abcm2ps musescore3 ghostscript imagemagick librsvg2-bin libmagickwand-dev
```

*   `abcm2ps`: Renders ABC notation to SVG.
*   `musescore3`: Optional, used for high-fidelity rendering if needed (currently using abcm2ps).
*   `librsvg2-bin`: Provides `rsvg-convert` for SVG-to-PNG conversion.
*   `imagemagick`: Graphics manipulation.
*   `libmagickwand-dev`: Headers for ImageMagick bindings.

### Python Environment

Install dependencies using `uv`:

```bash
uv sync 
# or manual install
uv pip install -r requirements.txt
```

## Running

1.  **Data Preparation**:
    Ensure your dataset is in ABC or MusicXML format.
    Use `gelato.data.converter` and `renderer` to prepare images.

2.  **Training**:
    Run the training script:
    ```bash
    uv run train.py --data_dir /path/to/data
    ```

    Example output:
    ```
    Done. Success: 174659, Skipped: 50004, Errors: 29372
    ```

    uv run train.py \                                                                                                                               HEAD
    --data_dir data/processed_train \
    --output_dir checkpoints/full_run_2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --fp16 \
    --gradient_checkpointing \
    --save_steps 500 \
    --max_seq_len 1024


| Stage | hands | fingerings | ornaments | dynamics | slurs | ties | time sigs |      script       | parameters |

|-------|-------|------------|-----------|----------|-------|------|-----------|-------------------|------------|

| 1     | 1     | ✗          | ✗         | ✗        | ✗     | ✗    | simple    | easy-writer    | data/dataset-stage-1/mxls   50000   |

| 2     | 1     | ✗          | ✗         | ✗        | ✗     | ✗    | simple    | inter-writer   | data/dataset-stage-2/mxls   25000   |

| 3     | 1     | ✗          | ✗         | ✗        | ✗     | ✗    | simple    | synthetic-writer   | data/dataset-stage-3/mxls   40000   --num_voices 1 --no_fingerings --no_ornaments --no_dynamics --no_slurs --no_ties --time_sig_set simple --scale_types major chord_progression --min_measures 4 --max_measures 12 |

| 4     | 1     | ✓          | ✗         | ✗        | ✗     | ✗    | medium    | synthetic-writer  | data/dataset-stage-4/mxls   50000   --num_voices 1 --no_ornaments --no_dynamics --no_slurs --no_ties --time_sig_set medium --min_measures 6 --max_measures 16

| 5     | 2     | ✓          | ✗         | ✓        | ✓     | ✗    | medium    | synthetic-writer | data/dataset-stage-5/mxls   80000   --no_ornaments --no_ties --time_sig_set medium --min_measures 8 --max_measures 20

| 6     | 2     | ✓          | ✓         | ✓        | ✓     | ✓    | full      |synthetic-writer | data/dataset-stage-6/mxls   100000   --no_ornaments --no_ties --time_sig_set medium --min_measures 8 --max_measures 20

