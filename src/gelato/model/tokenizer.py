"""
tokenizer_sync.py — Synchronized ABC tokenizer with gaps filled.

Changes from original tokenizer.py:
  - Added missing tokens: #, key modes (Dor, Mix, etc.), minor
  - Added tuplet marker (3 — handled by existing ( and digits
  - Added trill( and trill) span markers
  - Reordered vocab so multi-char tokens precede their prefixes
    (greedy matching picks the longest match first among special tokens)
  - Added : for V: attribute parsing (clef=bass etc.)
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


def build_abc_tokenizer(save_dir="./custom-abc-tokenizer"):
    # 1. Comprehensive ABC vocabulary
    # ORDER MATTERS for greedy matching: longer tokens first within groups
    abc_vocab = [
        # ── Control tokens ────────────────────────────────────────────
        "<pad>", "</s>", "<s>", "<unk>",

        # ── Whitespace ────────────────────────────────────────────────
        "\n", " ",

        # ── Headers (complete tokens) ────────────────────────────────
        "L:", "M:", "K:", "V:", "Q:", "P:",

        # ── Time signatures & header values ──────────────────────────
        "1/16", "1/8", "1/4",          # note lengths (longest first)
        "2/4", "3/4", "4/4", "6/8", "3/8",  # meters
        
        # ── Key signatures ───────────────────────────────────────────
        # Multi-char first so greedy matching works
        "Bb", "Eb", "Ab", "F#", "C#",
        # Modes (appear after key letter in K: lines)
        "Dor", "Mix", "Phr", "Lyd", "Loc",
        "minor",  # K:A minor — alternate form of minor key

        # ── Clef / voice values ──────────────────────────────────────
        "treble", "treble-8", "treble+8",
        "bass", "bass3",
        "alto", "alto1", "alto2", "alto4",
        "perc", "none",
        "clef=",        # V: attribute prefix
        "transpose=",   # V: transposition (transpose=-3)

        # ── Uppercase pitches / key names ────────────────────────────
        "C", "D", "E", "F", "G", "A", "B",

        # ── Lowercase pitches ────────────────────────────────────────
        "c", "d", "e", "f", "g", "a", "b",

        # ── Special notes ────────────────────────────────────────────
        "z", "Z", "X", "x", "y",

        # ── Accidentals (longest first) ──────────────────────────────
        "^^", "^", "__", "_", "=",

        # ── Octave modifiers (longest first) ─────────────────────────
        ",,", ",", "''", "'",

        # ── Digits & fraction ────────────────────────────────────────
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "/",

        # ── Barlines (longest first) ─────────────────────────────────
        "|]", "||", "|:", ":|", "::", "[1", "[2", "|",

        # ── Grouping & structure ─────────────────────────────────────
        "-", "(", ")", "[", "]", "{", "}", '"',

        # ── Combined decorations (single atomic tokens) ──────────────
        # Dynamics
        "!pppp!", "!ppp!", "!pp!", "!p!", "!mp!", "!mf!",
        "!f!", "!ff!", "!fff!", "!ffff!", "!sfz!",
        # Crescendo / diminuendo
        "!crescendo(!", "!<(!", "!crescendo)!", "!<)!",
        "!diminuendo(!", "!>(!", "!diminuendo)!", "!>)!",
        # Ornaments
        "!trill!", "!trill(!", "!trill)!",  # trill + trill span
        "!lowermordent!", "!uppermordent!", "!mordent!", "!pralltriller!",
        "!accent!", "!>!", "!emphasis!", "!fermata!", "!invertedfermata!",
        "!tenuto!",
        "!trem1!", "!trem2!", "!trem3!", "!trem4!",
        "!xstem!", "!slide!", "!turnx!", "!invertedturnx!",
        "!arpeggio!", "!invertedturn!",
        # Phrasing & fingering
        "!shortphrase!", "!mediumphrase!", "!longphrase!",
        "!upbow!", "!downbow!", "!thumb!", "!snap!",
        "!turn!", "!roll!", "!breath!",
        # Repeat/section
        "!segno!", "!coda!", "!D.S.!", "!D.C.!",
        "!dacoda!", "!dacapo!", "!fine!",
        # Fingering numbers
        "!0!", "!1!", "!2!", "!3!", "!4!", "!5!",
        # Extra
        "!plus!", "!wedge!", "!open!",
        # Bare boundary (fallback)
        "!",

        # ── Short-form decorations ───────────────────────────────────
        ".", "~",
        "H", "L", "M", "P", "S", "T", "u", "v", "O", "J", "R",

        # ── Chord symbol fragments ───────────────────────────────────
        "maj", "min", "dim", "aug", "sus",
        "m",    # minor shorthand
        "+",    # augmented shorthand
        "#",    # sharp in chord symbols (F#7, C#m)

        # ── Dotted ties/slurs ────────────────────────────────────────
        ".(", ".)",  ".-",
    ]

    # 2. Build BPE with the full vocab pre-loaded (no merges — every token
    #    is atomic). This ensures <unk> exists in the model vocabulary.
    vocab_dict = {tok: i for i, tok in enumerate(abc_vocab)}
    tokenizer = Tokenizer(BPE(vocab=vocab_dict, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = None  # special tokens handle all splitting

    # 3. Mark every token as "special" so it is NEVER split by BPE
    tokenizer.add_special_tokens(abc_vocab)

    # 4. BOS/EOS wrapping for generation
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", abc_vocab.index("<s>")),
            ("</s>", abc_vocab.index("</s>")),
        ],
    )

    # 5. Wrap for HuggingFace
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )

    if save_dir is not None:
        hf_tokenizer.save_pretrained(save_dir)
        print(f"Tokenizer saved to {save_dir}  |  vocab size: {len(hf_tokenizer)}")
    else:
        print(f"Tokenizer built dynamically |  vocab size: {len(hf_tokenizer)}")
        
    return hf_tokenizer


# ═══════════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tk = build_abc_tokenizer()

    tests = [
        # (input, expected_tokens)
        ("M:4/4",          ["M:", "4/4"]),
        ("K:G",            ["K:", "G"]),
        ("K:AMix",         ["K:", "A", "Mix"]),
        ("K:Bb",           ["K:", "Bb"]),
        ("!mf!",           ["!mf!"]),
        ("!trill!",        ["!trill!"]),
        ("!diminuendo(!",  ["!diminuendo(!"]),
        ("^^",             ["^^"]),
        ("__",             ["__"]),
        (",,",             [",,"]),
        ("''",             ["''"]),
        ("|]",             ["|]"]),
        ("||",             ["||"]),
        ("[1",             ["[1"]),
    ]

    print("\nTokenization tests:")
    all_pass = True
    for input_str, expected in tests:
        actual = tk.tokenize(input_str)
        status = "✓" if actual == expected else "✗"
        if actual != expected:
            all_pass = False
        print(f"  {status} '{input_str}' → {actual}  (expected {expected})")

    # Full-line test
    line = '!mf! "Am7" !trill!^c\'2 |] '
    tokens = tk.tokenize(line)
    ids = tk.encode(line, add_special_tokens=False)
    decoded = tk.decode(ids, skip_special_tokens=False)
    print(f"\n  Full line: {repr(line)}")
    print(f"  Tokens:    {tokens}")
    print(f"  IDs:       {ids}")
    print(f"  Decoded:   {repr(decoded)}")
    unk_id = tk.unk_token_id
    if unk_id in ids:
        print(f"  ⚠ WARNING: <unk> found at positions {[i for i,x in enumerate(ids) if x == unk_id]}")
    else:
        print(f"  ✓ No <unk> tokens")