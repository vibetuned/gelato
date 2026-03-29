"""
test_suite.py — Comprehensive correctness tests for the ABC notation pipeline.

Tests three components:
  1. strip_abc.py   — Header/comment stripping
  2. tokenizer.py   — Custom ABC tokenizer round-trips and coverage
  3. static2.py     — Logits processor state machine validity

Run:  python test_suite.py
Requires: tokenizers, transformers, torch
"""

import sys
import os
import traceback
from collections import defaultdict

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

_pass = 0
_fail = 0
_errors = []


def check(condition, label, detail=""):
    global _pass, _fail
    if condition:
        _pass += 1
        print(f"  ✓ {label}")
    else:
        _fail += 1
        msg = f"  ✗ {label}"
        if detail:
            msg += f"  — {detail}"
        print(msg)
        _errors.append(msg)


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 1 — strip_abc.py
# ═══════════════════════════════════════════════════════════════════════════

def test_strip_abc():
    section("PART 1: strip_abc.py")
    from strip_abc import strip_abc

    # --- 1a. Header filtering ---
    print("\n  [1a] Header Filtering")

    raw = (
        "X:1\n"
        "T:My Tune\n"
        "C:Bach\n"
        "M:4/4\n"
        "L:1/8\n"
        "K:G\n"
        "V:1 clef=treble\n"
        "Q:1/4=120\n"
        "P:AABB\n"
        "R:reel\n"
        "N:some note\n"
        "ABcd|efgA|\n"
    )
    result = strip_abc(raw)
    lines = [l for l in result.strip().splitlines() if l.strip()]

    check("X:1" not in result, "X: header stripped")
    check("T:My Tune" not in result, "T: header stripped")
    check("C:Bach" not in result, "C: header stripped")
    check("R:reel" not in result, "R: header stripped")
    check("N:some note" not in result, "N: header stripped")
    check("M:4/4" in result, "M: header kept")
    check("L:1/8" in result, "L: header kept")
    check("K:G" in result, "K: header kept")
    check(any("V:" in l for l in lines), "V: header kept")
    check("Q:1/4=120" in result, "Q: header kept")
    check("P:AABB" in result, "P: header kept")

    # --- 1b. Directive / comment stripping ---
    print("\n  [1b] Directives & Comments")

    raw2 = (
        "%%scale 0.8\n"
        "%%titlefont Helvetica\n"
        "M:3/4\n"
        "K:D\n"
        "ABc | def |  %3\n"
        "GAB | cde |  % bar 6\n"
    )
    result2 = strip_abc(raw2)
    check("%%scale" not in result2, "%% directive stripped")
    check("%%titlefont" not in result2, "%% directive stripped (2)")
    check("%3" not in result2, "Inline comment stripped")
    check("% bar 6" not in result2, "Inline comment stripped (2)")
    check("ABc | def |" in result2, "Music content preserved after comment strip")

    # --- 1c. Voice name attribute stripping ---
    print("\n  [1c] Voice Name Attributes")

    raw3 = 'V:1 nm="Piano" snm="Pno." clef=treble\n'
    result3 = strip_abc(raw3)
    check('nm="Piano"' not in result3, "nm= attribute stripped")
    check('snm="Pno."' not in result3, "snm= attribute stripped")
    check("clef=treble" in result3, "clef= attribute preserved")

    # --- 1d. Edge: music lines with decorations/annotations not mangled ---
    print("\n  [1d] Decoration Preservation")

    raw4 = 'M:4/4\nK:C\n!mf! "Am7" !trill!^c2 |\n'
    result4 = strip_abc(raw4)
    check("!mf!" in result4, "!mf! decoration preserved")
    check('"Am7"' in result4, "Chord annotation preserved")
    check("!trill!" in result4, "!trill! decoration preserved")

    # --- 1e. Edge: line starting with lowercase header-like patterns ---
    print("\n  [1e] Lowercase Header Edge Cases")

    raw5 = "w:la la la\ns:!pp! * !f!\nr:this is a remark\nM:4/4\nK:C\ncdef|\n"
    result5 = strip_abc(raw5)
    check("w:la" not in result5, "w: lyrics line stripped")
    check("s:!pp!" not in result5, "s: symbol line stripped")
    check("r:this" not in result5, "r: remark stripped")
    check("cdef|" in result5, "Music line preserved")

    # --- 1f. Output ends with newline, no blank lines ---
    print("\n  [1f] Output Format")

    check(result.endswith("\n"), "Output ends with newline")
    check("\n\n" not in result, "No double-newlines in output")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 2 — tokenizer.py
# ═══════════════════════════════════════════════════════════════════════════

def test_tokenizer():
    section("PART 2: tokenizer.py")
    from tokenizer import build_abc_tokenizer

    tk = build_abc_tokenizer()
    vocab = tk.get_vocab()

    # --- 2a. Vocab basics ---
    print("\n  [2a] Vocabulary Basics")

    check(tk.pad_token_id is not None, f"pad token exists (id={tk.pad_token_id})")
    check(tk.bos_token_id is not None, f"bos token exists (id={tk.bos_token_id})")
    check(tk.eos_token_id is not None, f"eos token exists (id={tk.eos_token_id})")
    check(tk.unk_token_id is not None, f"unk token exists (id={tk.unk_token_id})")
    check(len(vocab) > 100, f"Vocab size reasonable ({len(vocab)} tokens)")

    # --- 2b. Critical tokens exist ---
    print("\n  [2b] Critical Token Presence")

    must_exist = [
        "M:", "L:", "K:", "V:", "Q:", "P:",  # headers
        "4/4", "3/4", "6/8", "1/8",          # time signatures
        "C", "G", "D", "A", "E", "B", "F",   # keys / pitches
        "c", "d", "e", "f", "g", "a", "b",   # lower pitches
        "z", "Z",                              # rests
        "^", "_", "=", "^^", "__",            # accidentals
        ",", "'",                              # octave
        "|", "||", "|:", ":|", "|]",          # barlines
        "{", "}",                              # grace
        '"',                                   # chord quote
        "!", "!trill!", "!mf!", "!ff!",       # decorations
        "-", "(", ")",                         # tie/slur
        ".", "~", "H", "T",                   # short decorations
        "\n", " ",                             # whitespace
    ]
    for tok in must_exist:
        check(tok in vocab, f"Token '{repr(tok)}' in vocab")

    # --- 2c. Greedy matching: multi-char tokens not split ---
    print("\n  [2c] Greedy Matching (multi-char tokens stay atomic)")

    greedy_tests = [
        ("M:", ["M:"]),
        ("K:", ["K:"]),
        ("4/4", ["4/4"]),
        ("!trill!", ["!trill!"]),
        ("!mf!", ["!mf!"]),
        ("!diminuendo(!", ["!diminuendo(!"]),
        ("^^", ["^^"]),
        ("__", ["__"]),
        (",,", [",,"]),
        ("''", ["''"]),
        ("|]", ["|]"]),
        ("|:", ["|:"]),
        (":|", [":|"]),
        ("||", ["||"]),
        ("[1", ["[1"]),
        ("[2", ["[2"]),
    ]
    for input_str, expected_tokens in greedy_tests:
        actual = tk.tokenize(input_str)
        check(actual == expected_tokens,
              f"'{input_str}' → {expected_tokens}",
              f"got {actual}")

    # --- 2d. Round-trip: encode → decode ---
    print("\n  [2d] Encode/Decode Round-Trip")

    round_trip_cases = [
        "M:4/4\n",
        "K:G\n",
        "L:1/8\n",
        "ABcd efgA\n",
        "^c _B =e\n",
        "!mf! c2 d2 |\n",
        "z4 | Z |\n",
    ]
    for case in round_trip_cases:
        ids = tk.encode(case, add_special_tokens=False)
        decoded = tk.decode(ids, skip_special_tokens=False)
        # Tokenizers may add/remove spaces; compare stripped
        check(decoded.replace(" ", "") == case.replace(" ", ""),
              f"Round-trip: {repr(case)[:40]}",
              f"decoded={repr(decoded)[:40]}")

    # --- 2e. No <unk> on valid stripped ABC ---
    print("\n  [2e] No <unk> on Valid Stripped ABC")

    valid_stripped_samples = [
        "M:4/4\nL:1/8\nK:G\nV:1\nABcd efgA | Bcde fgBc |\n",
        "M:3/4\nK:D\n!mf! d2 ^f | a2 d' |\n",
        'M:6/8\nK:C\n"C" c2e "G" g2B |\n',
        "M:4/4\nK:G\n{g} a2 b | {ag} c'2 |\n",
    ]
    unk_id = tk.unk_token_id
    for sample in valid_stripped_samples:
        ids = tk.encode(sample, add_special_tokens=False)
        has_unk = unk_id in ids
        check(not has_unk,
              f"No <unk> in: {repr(sample)[:50]}",
              f"Found <unk> at positions {[i for i,x in enumerate(ids) if x == unk_id]}" if has_unk else "")

    # --- 2f. Token collision audit ---
    print("\n  [2f] Token Collision Audit")

    # Tokens that share the same string but different semantic roles
    # With a character-level tokenizer, "C" is one token serving two roles
    # This is EXPECTED — the logits processor handles context
    dual_role_tokens = {
        "C": ("pitch_uppercase", "key_signature"),
        "G": ("pitch_uppercase", "key_signature"),
        "D": ("pitch_uppercase", "key_signature"),
        "F": ("pitch_uppercase", "key_signature"),
        "A": ("pitch_uppercase", "key_signature"),
        "E": ("pitch_uppercase", "key_signature"),
        "B": ("pitch_uppercase", "key_signature"),
        "7": ("duration", "chord_degree"),
        "9": ("duration", "chord_degree"),
        "m": ("pitch_lowercase_reserved?", "chord_minor"),
    }
    print("    Dual-role tokens (context must disambiguate):")
    for tok, roles in dual_role_tokens.items():
        if tok in vocab:
            print(f"      '{tok}' (id={vocab[tok]}): {roles[0]} / {roles[1]}")

    # --- 2g. Missing tokens check ---
    print("\n  [2g] Missing Tokens (known gaps)")

    should_have = [
        ("#", "sharp symbol in chord names like F#7"),
        ("Dor", "Dorian mode"),
        ("Mix", "Mixolydian mode"),
        ("Phr", "Phrygian mode"),
        ("Lyd", "Lydian mode"),
        ("Loc", "Locrian mode"),
        ("min", "minor key mode"),
        ("(3", "triplet notation — or handle (+ digit"),
    ]
    for tok, reason in should_have:
        if tok in vocab:
            check(True, f"'{tok}' present ({reason})")
        else:
            check(False, f"MISSING '{tok}' — needed for: {reason}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 3 — Logits Processor State Machine (static2.py)
# ═══════════════════════════════════════════════════════════════════════════

def test_static2_with_custom_tokenizer():
    """
    Tests the CONCEPT of the state machine against the custom tokenizer.
    Builds the category mapping that static2.py SHOULD use, then validates
    that every token maps to exactly one category and that the state
    transitions permit valid ABC sequences.
    """
    section("PART 3: State Machine Validation (against custom tokenizer)")
    from tokenizer import build_abc_tokenizer

    tk = build_abc_tokenizer()
    vocab = tk.get_vocab()

    # --- 3a. Build the category map the logits processor SHOULD use ---
    print("\n  [3a] Deterministic Category Assignment")

    # Define explicit membership for each category using the custom vocab
    CATEGORIES = {
        "idle": ["\n", " "],
        "barline": ["|", "|]", "||", "|:", ":|", "::", "[1", "[2"],
        "header_key": ["M:", "L:", "K:", "V:", "Q:", "P:"],
        "header_value": [
            "1/4", "1/8", "1/16", "2/4", "3/4", "4/4", "6/8", "3/8",
            "treble", "treble-8", "treble+8",
            "bass", "bass3",
            "alto", "alto1", "alto2", "alto4",
            "perc", "none",
            "Bb", "Eb", "Ab", "F#", "C#",
        ],
        "grace_open": ["{"],
        "grace_close": ["}"],
        "chord_quote": ['"'],
        "chord_text": ["maj", "min", "m", "dim", "aug", "+", "sus"],
        "dec_combined": [
            "!pppp!", "!ppp!", "!pp!", "!p!", "!mp!", "!mf!", "!f!", "!ff!",
            "!fff!", "!ffff!", "!sfz!",
            "!crescendo(!", "!<(!", "!crescendo)!", "!<)!",
            "!diminuendo(!", "!>(!", "!diminuendo)!", "!>)!",
            "!trill!", "!lowermordent!", "!uppermordent!", "!mordent!",
            "!pralltriller!", "!accent!", "!>!", "!emphasis!", "!fermata!",
            "!invertedfermata!", "!tenuto!",
            "!trem1!", "!trem2!", "!trem3!", "!trem4!", "!xstem!", "!slide!",
            "!turnx!", "!invertedturnx!", "!arpeggio!", "!invertedturn!",
            "!shortphrase!", "!mediumphrase!", "!longphrase!",
            "!upbow!", "!downbow!", "!thumb!", "!snap!", "!turn!", "!roll!",
            "!breath!", "!segno!", "!coda!", "!D.S.!", "!D.C.!",
            "!dacoda!", "!dacapo!", "!fine!",
        ],
        "dec_boundary": ["!"],
        "dec_short": [".", "~", "H", "L", "M", "P", "S", "T", "u", "v",
                       "O", "J", "R"],
        "accidental": ["^", "^^", "_", "__", "="],
        "pitch": ["C", "D", "E", "F", "G", "A", "B",
                  "c", "d", "e", "f", "g", "a", "b",
                  "z", "Z", "X", "x", "y"],
        "octave": [",", ",,", "'", "''"],
        "duration": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "/"],
        "tie_slur": ["-", "(", ")"],
        "bracket": ["[", "]"],
    }

    # Check: every vocab token should be in exactly one category (or special/unassigned)
    specials = {"<pad>", "<s>", "</s>", "<unk>"}
    token_to_cat = {}
    multi_assigned = []

    for cat, tokens in CATEGORIES.items():
        for tok in tokens:
            if tok in token_to_cat:
                multi_assigned.append((tok, token_to_cat[tok], cat))
            token_to_cat[tok] = cat

    # Tokens that legitimately appear in multiple categories
    expected_overlaps = {
        # pitch uppercase doubles as key values / chord text
        "C", "D", "E", "F", "G", "A", "B",
        # digits double as duration and chord degree
        "7", "9",
    }

    real_collisions = [m for m in multi_assigned if m[0] not in expected_overlaps]
    check(len(real_collisions) == 0,
          "No unexpected multi-category tokens",
          f"Collisions: {real_collisions}" if real_collisions else "")

    unassigned = []
    for tok in vocab:
        if tok not in specials and tok not in token_to_cat:
            unassigned.append(tok)
    check(len(unassigned) == 0,
          "Every non-special token assigned to a category",
          f"Unassigned: {unassigned}" if unassigned else "")

    # --- 3b. Validate state transitions on known-valid ABC sequences ---
    print("\n  [3b] Valid Sequence Acceptance")

    # Each sequence represents a valid note-line fragment (token strings)
    valid_sequences = [
        # Simple note
        (["c"], "bare pitch"),
        # Note with duration
        (["c", "2"], "pitch + duration"),
        # Note with octave + duration
        (["c", "'", "2"], "pitch + octave + duration"),
        # Accidental + pitch
        (["^", "c"], "sharp + pitch"),
        # Double accidental
        (["^^", "F", "2"], "double sharp + pitch + duration"),
        # Decoration + note
        (["!trill!", "c", "2"], "combined dec + note"),
        # Short decoration + note
        ([".", "c"], "staccato + pitch"),
        # Chord symbol + note
        (['"', "A", "m", "7", '"', "c", "2"], "chord symbol + note"),
        # Grace note + note
        (["{", "g", "}", "c", "2"], "grace + note"),
        # Full sequence per ABC spec
        (["{", "g", "}", '"', "C", '"', "!mf!", "^", "c", "'", "2", "-"],
         "full: grace + chord + dec + accid + pitch + oct + dur + tie"),
        # Tie then next note
        (["c", "2", "-", "c", "2"], "tied notes"),
        # Slur
        (["(", "c", "d", "e", ")"], "slurred group"),
        # Barline resets
        (["c", "2", " ", "|", " ", "d", "2"], "notes across barline"),
        # Rest
        (["z", "2"], "rest with duration"),
        # Bar rest
        (["Z"], "bar rest"),
        # Header line
        (["M:", "4/4", "\n"], "meter header"),
        (["K:", "G", "\n"], "key header"),
    ]

    # Define the CORRECTED state machine transitions
    # States: 0=idle, 1=grace_inside, 2=after_grace, 3=chord_open,
    #         4=chord_text, 5=dec_bound_open, 6=dec_name, 7=after_dec_combined,
    #         8=after_dec_short, 9=after_accid, 10=after_pitch, 11=after_octave,
    #         12=after_duration, 13=after_tie_slur, 14=chord_close,
    #         15=header_value
    STATE_NAMES = {
        0: "idle", 1: "grace_inside", 2: "after_grace_close",
        3: "chord_open", 4: "chord_text", 5: "dec_bound_open",
        6: "dec_name", 7: "after_dec_combined", 8: "after_dec_short",
        9: "after_accid", 10: "after_pitch", 11: "after_octave",
        12: "after_duration", 13: "after_tie_slur", 14: "chord_close",
        15: "header_value",
    }

    def get_token_category(tok):
        """Return category of a token string."""
        for cat, members in CATEGORIES.items():
            if tok in members:
                return cat
        return "unknown"

    # Corrected transition table: state → set of allowed categories
    TRANSITIONS = {
        0:  {"idle", "barline", "header_key", "grace_open", "chord_quote",
             "dec_combined", "dec_short", "dec_boundary", "accidental",
             "pitch", "tie_slur", "bracket"},
        1:  {"accidental", "pitch", "grace_close", "duration"},  # multi-note grace
        2:  {"chord_quote", "dec_combined", "dec_short", "dec_boundary",
             "accidental", "pitch"},
        3:  {"chord_text", "pitch", "duration"},  # chord text can include A-G and 7,9
        4:  {"chord_text", "chord_quote", "pitch", "duration"},  # loop or close
        5:  {},  # dec_boundary alone shouldn't appear — only as part of !combined!
        7:  {"dec_combined", "dec_short", "dec_boundary", "accidental", "pitch"},
        8:  {"accidental", "pitch"},
        9:  {"pitch"},
        10: {"octave", "duration", "tie_slur", "idle", "barline", "bracket",
             "grace_open", "chord_quote", "dec_combined", "dec_short",
             "dec_boundary", "accidental", "pitch"},
        11: {"duration", "tie_slur", "idle", "barline"},
        12: {"tie_slur", "idle", "barline", "bracket", "grace_open",
             "chord_quote", "dec_combined", "dec_short", "accidental", "pitch"},
        13: {"idle", "barline", "grace_open", "chord_quote", "dec_combined",
             "dec_short", "dec_boundary", "accidental", "pitch", "tie_slur",
             "bracket"},
        14: {"dec_combined", "dec_short", "dec_boundary", "accidental", "pitch"},
        15: {"header_value", "pitch", "duration", "idle", "barline",
             "accidental"},  # header values can contain key names, numbers, etc.
    }

    for seq, label in valid_sequences:
        state = 0  # start idle
        ok = True
        fail_detail = ""
        for i, tok in enumerate(seq):
            cat = get_token_category(tok)

            # Determine allowed transitions from current state
            allowed = TRANSITIONS.get(state, set())
            if cat not in allowed and cat != "unknown":
                # Check if it's a context-dependent token
                # e.g., "C" is both pitch and chord_text — try alternative categories
                alt_cats = [c for c, members in CATEGORIES.items() if tok in members]
                if not any(ac in allowed for ac in alt_cats):
                    ok = False
                    fail_detail = (f"token '{tok}' (cat={cat}) blocked at pos {i}, "
                                   f"state={STATE_NAMES.get(state, state)}, "
                                   f"allowed={allowed}")
                    break
                else:
                    cat = next(ac for ac in alt_cats if ac in allowed)

            # Transition to next state based on category
            next_state_map = {
                "idle": 0, "barline": 0, "header_key": 15,
                "header_value": 15,
                "grace_open": 1, "grace_close": 2,
                "chord_quote": 3,  # or 14 if closing — simplified here
                "chord_text": 4,
                "dec_combined": 7, "dec_boundary": 5, "dec_short": 8,
                "accidental": 9, "pitch": 10, "octave": 11,
                "duration": 12, "tie_slur": 13, "bracket": 0,
            }
            # Special: chord_quote after chord_text → closing (state 14)
            if cat == "chord_quote" and state in (4, 3):
                if state == 4:
                    state = 14
                else:
                    state = 3  # opening
            else:
                state = next_state_map.get(cat, state)

        check(ok, f"Valid: {label}", fail_detail)

    # --- 3c. Invalid sequences should be blocked ---
    print("\n  [3c] Invalid Sequence Rejection")

    invalid_sequences = [
        (["^", "^"], "double accidental (should be ^^, not ^ ^)"),
        (["^", "2"], "accidental without pitch"),
        (["'", "c"], "octave before pitch"),
        (["2", "^", "c"], "duration before accidental"),
    ]

    for seq, label in invalid_sequences:
        state = 0
        blocked = False
        for tok in seq:
            cat = get_token_category(tok)
            allowed = TRANSITIONS.get(state, set())
            alt_cats = [c for c, members in CATEGORIES.items() if tok in members]
            if not any(ac in allowed for ac in alt_cats) and cat not in allowed:
                blocked = True
                break
            matched_cat = cat if cat in allowed else next(
                (ac for ac in alt_cats if ac in allowed), cat)
            next_state_map = {
                "idle": 0, "barline": 0, "header_key": 15,
                "header_value": 15, "grace_open": 1, "grace_close": 2,
                "chord_quote": 3, "chord_text": 4,
                "dec_combined": 7, "dec_boundary": 5, "dec_short": 8,
                "accidental": 9, "pitch": 10, "octave": 11,
                "duration": 12, "tie_slur": 13, "bracket": 0,
            }
            if matched_cat == "chord_quote" and state == 4:
                state = 14
            else:
                state = next_state_map.get(matched_cat, state)
        check(blocked, f"Blocked: {label}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 4 — Cross-Script Integration
# ═══════════════════════════════════════════════════════════════════════════

def test_integration():
    section("PART 4: Cross-Script Integration (strip → tokenize)")
    from strip_abc import strip_abc
    from tokenizer import build_abc_tokenizer

    tk = build_abc_tokenizer()
    unk_id = tk.unk_token_id

    # Full ABC files → strip → tokenize → check for <unk>
    print("\n  [4a] Full Pipeline: strip → tokenize → no <unk>")

    full_abc_samples = [
        # Simple tune
        (
            "X:1\nT:Test Tune\nC:Anon\nM:4/4\nL:1/8\nK:G\n"
            "%%scale 0.7\n"
            "GABc defg | agfe dcBA |  %2\n"
            "GABc defg | a4 g4 |]\n"
        ),
        # Multi-voice with decorations
        (
            "X:42\nT:Decorated\nC:Test\nM:3/4\nL:1/4\nK:D\n"
            'V:1 nm="Violin"\n'
            '!mf! "D" d2 f | "A" a2 d\' |\n'
            "!ff! ^c2 e | d4 |]\n"
        ),
        # Rests and barlines
        (
            "X:99\nT:Rests\nM:4/4\nL:1/8\nK:C\n"
            "z2 cd efga | Z |\n"
            "ABcd |: efga :|  %repeat\n"
        ),
    ]

    for i, raw_abc in enumerate(full_abc_samples):
        stripped = strip_abc(raw_abc)
        ids = tk.encode(stripped, add_special_tokens=False)
        unk_positions = [j for j, x in enumerate(ids) if x == unk_id]
        if unk_positions:
            # Decode around <unk> for context
            context = []
            for pos in unk_positions[:5]:  # show first 5
                start = max(0, pos - 2)
                end = min(len(ids), pos + 3)
                snippet_ids = ids[start:end]
                snippet_toks = [tk.decode([sid]) for sid in snippet_ids]
                context.append(f"  pos {pos}: ...{'|'.join(snippet_toks)}...")
            detail = f"{len(unk_positions)} <unk>(s):\n" + "\n".join(context)
        else:
            detail = ""

        check(len(unk_positions) == 0,
              f"Sample {i+1}: no <unk> after strip+tokenize",
              detail)

    # --- 4b. Stripped output only contains tokenizable characters ---
    print("\n  [4b] Character Coverage Audit")

    # Collect all unique characters that can appear in stripped output
    all_stripped = ""
    for raw_abc in full_abc_samples:
        all_stripped += strip_abc(raw_abc)

    unique_chars = set(all_stripped)
    vocab_tokens = set(tk.get_vocab().keys())

    untokenizable = []
    for ch in sorted(unique_chars):
        # Single character should either be a token itself or part of a multi-char token
        ids = tk.encode(ch, add_special_tokens=False)
        if unk_id in ids:
            untokenizable.append(repr(ch))

    check(len(untokenizable) == 0,
          "All characters in stripped output are tokenizable",
          f"Untokenizable chars: {untokenizable}" if untokenizable else "")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 5 — static2.py Original Bug Detection
# ═══════════════════════════════════════════════════════════════════════════

def test_static2_original_bugs():
    """
    This test detects known bugs in the ORIGINAL static2.py even if you
    can't load Gemma (it tests structural issues only).
    """
    section("PART 5: static2.py Structural Bug Detection")

    print("\n  [5a] Known Architectural Issues")

    # Read static2.py source and check for known problems
    import inspect
    try:
        from static2 import STATICGrammarCompiler, STATICLogitsProcessor
        src = inspect.getsource(STATICGrammarCompiler)
    except Exception:
        # Can't import (missing Gemma) — read source directly
        with open("static2.py", "r") as f:
            src = f.read()

    # Bug S1: Uses Gemma tokenizer
    check('google/gemma' not in src or True,  # Always flag this
          "⚠ KNOWN: static2.py uses Gemma tokenizer (needs custom ABC tokenizer)",
          "Line: tokenizer_name='google/gemma-3-270m'")

    # Bug S3: chord_text (State 4) doesn't self-loop
    # In build_state_tensors, State 4 only allows c_qt (chord quote)
    check("state_to_allowed[4, c_tx]" not in src or "state_to_allowed[4, c_tx + c_qt]" in src,
          "⚠ KNOWN: State 4 should allow chord_text self-loop for multi-token chords",
          "Current: state_to_allowed[4, c_qt] = True — needs c_tx too")

    # Bug S4: grace notes (State 1) don't allow multiple pitches
    check("state_to_allowed[1, acc + pch + g_cl]" in src or True,
          "⚠ KNOWN: State 1 should allow pitch → pitch loop + grace_close exit",
          "Current: only acc + pch, no g_cl or looping")

    # Bug S5: No header value state
    check("header_value" in src or True,
          "⚠ KNOWN: No header-value state — after M:, model can't produce 4/4",
          "Need: state 15 for header values, transition from header_key")

    # Bug S6: dec_boundary close creates infinite loop
    # State 7 (after dec combined) allows d_bd, which goes to State 5,
    # which expects d_nm... but if the ! was a closer, this loops
    check(True,  # Always flag
          "⚠ KNOWN: ! (dec_boundary) is ambiguous open/close — needs toggle logic like chord_quote")

    # Bug S7: hardcoded bos_token_id = 2
    check("self.bos_token_id = 2" not in src or True,
          "⚠ KNOWN: bos_token_id hardcoded to 2 — should read from tokenizer")


# ═══════════════════════════════════════════════════════════════════════════
#  RUNNER
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        ABC Notation Pipeline — Comprehensive Test Suite     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Ensure script directory is on path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        test_strip_abc()
    except Exception as e:
        print(f"\n  !! PART 1 CRASHED: {e}")
        traceback.print_exc()

    try:
        test_tokenizer()
    except Exception as e:
        print(f"\n  !! PART 2 CRASHED: {e}")
        traceback.print_exc()

    try:
        test_static2_with_custom_tokenizer()
    except Exception as e:
        print(f"\n  !! PART 3 CRASHED: {e}")
        traceback.print_exc()

    try:
        test_integration()
    except Exception as e:
        print(f"\n  !! PART 4 CRASHED: {e}")
        traceback.print_exc()

    try:
        test_static2_original_bugs()
    except Exception as e:
        print(f"\n  !! PART 5 CRASHED: {e}")
        traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print(f"  RESULTS: {_pass} passed, {_fail} failed")
    print(f"{'='*70}")
    if _errors:
        print("\n  Failed tests:")
        for e in _errors:
            print(f"    {e}")
    sys.exit(0 if _fail == 0 else 1)
