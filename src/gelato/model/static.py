"""
static2_sync.py — Logits processor synchronized with the custom ABC tokenizer.

Changes from original static2.py:
  - Uses the custom ABC tokenizer (not Gemma)
  - Deterministic category assignment (no regex guessing)
  - Fixes: chord-text self-loop, multi-note grace, header-value state,
    dec-boundary open/close toggle, bracket chords, tuplets
  - bos_token_id read from tokenizer
"""

import torch
from transformers import LogitsProcessor


class ABCGrammarCompiler:
    """
    Builds state-machine tensors for constrained ABC generation.

    Designed for the custom tokenizer from tokenizer.py — every token maps
    to exactly one grammar category via an explicit lookup table (no regex).
    """

    UNCATEGORIZED_STATE = 17  # Tokens not in any category land here

    def __init__(self, tokenizer):
        """
        Args:
            tokenizer: a HuggingFace PreTrainedTokenizerFast built by
                       tokenizer.py's build_abc_tokenizer().
        """
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()
        self.vocab_size = len(tokenizer)

    # ------------------------------------------------------------------
    # Category definitions — every non-special token is assigned here
    # ------------------------------------------------------------------

    # Decoration names used inside !...! boundaries (the inner text)
    # With the custom tokenizer these are ALWAYS single combined tokens
    # like !trill!, so we don't need a separate "dec_name" list.

    _CATEGORY_TABLE = {
        # ── Structural ────────────────────────────────────────────────
        "idle":         ["\n", " "],
        "barline":      ["|", "|]", "||", "|:", ":|", "::", "[1", "[2"],
        "header_key":   ["M:", "L:", "K:", "V:", "Q:", "P:"],
        "header_value": [
            "1/4", "1/8", "1/16", "2/4", "3/4", "4/4", "6/8", "3/8",
            "treble", "treble-8", "treble+8",
            "bass", "bass3",
            "alto", "alto1", "alto2", "alto4",
            "perc", "none",
            "Bb", "Eb", "Ab", "F#", "C#",
            "Dor", "Mix", "Phr", "Lyd", "Loc",  # key modes
            "minor",
            "clef=",        # V: attribute prefix
            "transpose=",   # V: transposition (transpose=-3)
        ],
        # ── Sequence modifiers ────────────────────────────────────────
        "grace_open":   ["{"],
        "grace_close":  ["}"],
        "chord_quote":  ['"'],
        "chord_text":   ["maj", "min", "m", "dim", "aug", "+", "sus",
                         "#",                          # sharp in chord names (F#7)
                         "Bb", "Eb", "Ab", "F#", "C#", # flat/sharp roots in chords
                        ],
        # ── Decorations ──────────────────────────────────────────────
        # Combined tokens: these are the ONLY decoration forms in the
        # custom tokenizer (no bare ! + name + ! sequence needed)
        "dec_combined":  [
            "!pppp!", "!ppp!", "!pp!", "!p!", "!mp!", "!mf!", "!f!", "!ff!",
            "!fff!", "!ffff!", "!sfz!",
            "!crescendo(!", "!<(!", "!crescendo)!", "!<)!",
            "!diminuendo(!", "!>(!", "!diminuendo)!", "!>)!",
            "!trill!", "!trill(!", "!trill)!",
            "!lowermordent!", "!uppermordent!", "!mordent!",
            "!pralltriller!", "!accent!", "!>!", "!emphasis!", "!fermata!",
            "!invertedfermata!", "!tenuto!",
            "!trem1!", "!trem2!", "!trem3!", "!trem4!", "!xstem!", "!slide!",
            "!turnx!", "!invertedturnx!", "!arpeggio!", "!invertedturn!",
            "!shortphrase!", "!mediumphrase!", "!longphrase!",
            "!upbow!", "!downbow!", "!thumb!", "!snap!", "!turn!", "!roll!",
            "!breath!", "!segno!", "!coda!", "!D.S.!", "!D.C.!",
            "!dacoda!", "!dacapo!", "!fine!",
            "!0!", "!1!", "!2!", "!3!", "!4!", "!5!",  # fingering
            "!plus!", "!wedge!", "!open!",
        ],
        "dec_boundary": ["!"],  # bare ! — kept for backward compat but
                                # should rarely appear with the custom tokenizer
        "dec_short":    [".", "~", "H", "L", "M", "P", "S", "T", "u", "v",
                         "O", "J", "R"],
        # ── Note elements ────────────────────────────────────────────
        "accidental":   ["^", "^^", "_", "__", "="],
        "pitch":        ["C", "D", "E", "F", "G", "A", "B",
                         "c", "d", "e", "f", "g", "a", "b",
                         "z", "Z", "X", "x", "y"],
        "octave":       [",", ",,", "'", "''"],
        "duration":     ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "/"],
        "tie_slur":     ["-", "(", ")", ".(", ".)", ".-"],
        "bracket":      ["[", "]"],
    }

    # Tokens that legitimately belong to multiple categories.
    # The state machine resolves ambiguity at runtime.
    _MULTI_ROLE = {
        # Uppercase pitches double as key-signature values and chord roots
        "C", "D", "E", "F", "G", "A", "B",
        # Digits 7, 9 are both durations and chord degrees
        "7", "9",
        # "+" is both chord-aug symbol and a standalone token
        "+",
        # Flat/sharp keys are both header values and chord roots
        "Bb", "Eb", "Ab", "F#", "C#",
    }

    def _build_token_categories(self):
        """
        Returns:
            primary: dict[str, list[int]]  — category name → list of token IDs
            token_cats: dict[int, set[str]] — token ID → set of category names

        Iterates over every token in the vocab, decodes it, and matches
        against the category table. This is tokenizer-agnostic — works
        regardless of how the vocab keys are stored internally (sentencepiece
        byte-level, BPE Ġ-prefixed, raw strings, etc.).
        """
        primary = {cat: [] for cat in self._CATEGORY_TABLE}
        token_cats = {}  # token_id → set of categories

        # Build reverse lookup: decoded_string → list of (category_name,)
        # We index both raw and stripped forms for whitespace tokens
        _lookup_raw = {}     # matches against raw decoded (preserves whitespace)
        _lookup_clean = {}   # matches against stripped decoded

        for cat, token_strs in self._CATEGORY_TABLE.items():
            for tok_str in token_strs:
                _lookup_raw.setdefault(tok_str, []).append(cat)
                stripped = tok_str.strip()
                if stripped and stripped != tok_str:
                    # Also index the stripped form (for tokens whose category
                    # string has no whitespace but whose decoded form might)
                    _lookup_clean.setdefault(stripped, []).append(cat)
                elif stripped:
                    _lookup_clean.setdefault(stripped, []).append(cat)

        # Scan every token in the vocab
        for token_str, token_id in self.vocab.items():
            raw_decoded = self.tokenizer.decode([token_id])
            clean_decoded = raw_decoded.strip()

            matched_cats = set()

            # 1. Try raw match first (catches \n, space, etc.)
            if raw_decoded in _lookup_raw:
                matched_cats.update(_lookup_raw[raw_decoded])

            # 2. Try clean match (catches everything else)
            if clean_decoded in _lookup_clean:
                matched_cats.update(_lookup_clean[clean_decoded])

            # Assign to categories
            for cat in matched_cats:
                primary[cat].append(token_id)
                token_cats.setdefault(token_id, set()).add(cat)

        return primary, token_cats

    def build_state_tensors(self, device="cpu"):
        """
        Returns:
            token_to_primary_state: LongTensor[vocab_size]
            state_to_allowed: BoolTensor[num_states, vocab_size]
            token_extra_cats: dict[int, set[str]]  — for runtime disambiguation
        """
        cats, token_cats = self._build_token_categories()

        # ── States ────────────────────────────────────────────────────
        #  0  idle          (whitespace, barlines, start)
        #  1  grace_inside  (after '{', inside grace notes)
        #  2  after_grace   (after '}')
        #  3  chord_open    (after opening '"')
        #  4  chord_text    (inside chord name, can loop)
        #  5  dec_bound_open (after opening '!' — expect dec name)
        #  6  dec_name      (after decoration name inside ! — expect closing !)
        #  7  after_dec     (after !...! combined or after dec close)
        #  8  after_dec_short (after ., ~, H, etc.)
        #  9  after_accid   (after ^, _, =, etc.)
        # 10  after_pitch   (after note letter)
        # 11  after_octave  (after , or ')
        # 12  after_duration (after digit or /)
        # 13  after_tie_slur (after -, (, ))
        # 14  chord_close   (after closing '"')
        # 15  header_value  (after M:, K:, etc. — free until \n)
        # 16  bracket_open  (after [ for inline chords [ceg])
        # 17  UNCATEGORIZED (default — blocks everything except EOS/PAD)
        NUM_STATES = 18
        UNCATEGORIZED_STATE = 17

        # Initialize ALL tokens to uncategorized — only categorized tokens
        # get overwritten below. This distinguishes "idle by design" (state 0)
        # from "not in any category" (state 17).
        token_to_state = torch.full(
            (self.vocab_size,), UNCATEGORIZED_STATE,
            dtype=torch.long, device=device
        )

        # Primary state assignment — last write wins for multi-role tokens,
        # but the logits processor uses state_to_allowed which includes
        # ALL roles via the allowed-mask.
        state_map = {
            "idle": 0, "barline": 0, "header_key": 15,
            "header_value": 15,
            "grace_open": 1, "grace_close": 2,
            "chord_quote": 3, "chord_text": 4,
            "dec_boundary": 5, "dec_combined": 7,
            "dec_short": 8, "accidental": 9,
            "pitch": 10, "octave": 11,
            "duration": 12, "tie_slur": 13,
            "bracket": 16,
        }

        for cat, state_id in state_map.items():
            for tid in cats.get(cat, []):
                token_to_state[tid] = state_id

        # ── Allowed-token masks per state ─────────────────────────────
        S = torch.zeros((NUM_STATES, self.vocab_size), dtype=torch.bool, device=device)

        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id

        # Helper: flatten multiple categories into a single ID list
        def ids(*cat_names):
            out = []
            for c in cat_names:
                out.extend(cats.get(c, []))
            return out

        # State 0 — Idle: full freedom to start any element
        S[0, ids("idle", "barline", "header_key", "grace_open", "chord_quote",
                 "dec_combined", "dec_short", "accidental", "pitch",
                 "tie_slur", "bracket")] = True

        # State 1 — Inside grace {}: allow accidentals, pitches (loop), 
        #   durations (for {/g} acciaccatura), and close
        S[1, ids("accidental", "pitch", "duration", "grace_close")] = True

        # State 2 — After grace close }: forward to chord, dec, accid, pitch
        S[2, ids("chord_quote", "dec_combined", "dec_short",
                 "accidental", "pitch")] = True

        # State 3 — After opening ": expect chord text tokens
        #   (pitch letters serve as chord roots: C, D, Am, etc.)
        S[3, ids("chord_text", "pitch", "duration", "accidental")] = True

        # State 4 — Inside chord text: continue chord or close quote
        S[4, ids("chord_text", "chord_quote", "pitch", "duration",
                 "accidental")] = True

        # State 5 — After opening ! (bare): expect decoration name
        # With the custom tokenizer this state is rarely reached because
        # decorations are single combined tokens. But if bare ! exists:
        # we'd need dec_name tokens. For safety, allow anything that
        # could be part of a decoration name.
        S[5, ids("dec_short", "dec_boundary")] = True  # minimal fallback

        # State 6 — After dec name: expect closing !
        S[6, ids("dec_boundary")] = True

        # State 7 — After combined decoration (!trill!, !mf!, etc.):
        #   can chain more decorations, or proceed to accid/pitch
        S[7, ids("dec_combined", "dec_short", "accidental", "pitch")] = True

        # State 8 — After short decoration (., ~, H, etc.): → accid or pitch
        S[8, ids("accidental", "pitch")] = True

        # State 9 — After accidental: MUST produce a pitch
        S[9, ids("pitch")] = True

        # State 10 — After pitch: octave, duration, tie, or end group
        S[10, ids("octave", "duration", "tie_slur", "idle", "barline",
                  "grace_open", "chord_quote", "dec_combined", "dec_short",
                  "accidental", "pitch", "bracket")] = True

        # State 11 — After octave: duration, tie, or end group
        S[11, ids("duration", "tie_slur", "idle", "barline",
                  "pitch", "accidental", "dec_combined", "dec_short",
                  "grace_open", "chord_quote", "bracket")] = True

        # State 12 — After duration: tie, or end group, or more duration digits
        S[12, ids("duration", "tie_slur", "idle", "barline",
                  "grace_open", "chord_quote", "dec_combined", "dec_short",
                  "accidental", "pitch", "bracket")] = True

        # State 13 — After tie/slur: can start next note group or idle
        S[13, ids("idle", "barline", "grace_open", "chord_quote",
                  "dec_combined", "dec_short", "accidental", "pitch",
                  "tie_slur", "bracket")] = True

        # State 14 — After closing chord quote: → dec, accid, pitch
        S[14, ids("dec_combined", "dec_short", "accidental", "pitch",
                  "idle", "barline")] = True

        # State 15 — Header value (after M:, K:, etc.): allow header values,
        #   key names (pitch tokens), digits, accidentals, idle (newline ends it),
        #   tie_slur (for negative numbers in transpose=-3)
        S[15, ids("header_value", "pitch", "duration", "idle", "barline",
                  "accidental", "tie_slur")] = True

        # State 16 — Inside bracket chord [ceg]: pitches, accidentals,
        #   durations, and close bracket
        S[16, ids("pitch", "accidental", "duration", "octave", "bracket")] = True

        # State 17 — UNCATEGORIZED: no transitions allowed (only EOS/PAD
        # from the block above). Any token that decodes to something not in
        # _CATEGORY_TABLE lands here and is effectively blocked during generation.

        # EOS/PAD always allowed
        if eos is not None:
            S[:, eos] = True
        if pad is not None:
            S[:, pad] = True

        # ── Coverage diagnostic ───────────────────────────────────────
        n_uncat = (token_to_state == UNCATEGORIZED_STATE).sum().item()
        n_categorized = self.vocab_size - n_uncat
        if n_uncat > 0:
            print(f"  Grammar coverage: {n_categorized}/{self.vocab_size} tokens "
                  f"categorized, {n_uncat} uncategorized (state {UNCATEGORIZED_STATE})")

        return token_to_state, S, token_cats


class ABCLogitsProcessor(LogitsProcessor):
    """
    Constrains generation to follow valid ABC note-element ordering.

    Uses a state machine with dynamic disambiguation for:
      - chord quotes (open vs close via previous-token context)
      - bracket chords ([ open vs ] close)
      - header lines (free-form values until newline)
    """

    def __init__(self, token_to_state, state_to_allowed, tokenizer):
        self.token_to_state = token_to_state
        self.state_to_allowed = state_to_allowed
        self.bos_token_id = tokenizer.bos_token_id or 1
        self.eos_token_id = tokenizer.eos_token_id or 2

        # Helper: resolve a token string → single token ID via encode(),
        # bypassing BPE internal key encoding mismatches.
        def _resolve(tok_str):
            ids = tokenizer.encode(tok_str, add_special_tokens=False)
            return ids[0] if len(ids) == 1 else None

        # Pre-compute token IDs for dynamic state adjustments
        self._quote_id = _resolve('"')
        self._newline_id = _resolve('\n')
        self._bracket_open_id = _resolve('[')
        self._bracket_close_id = _resolve(']')
        self._grace_close_id = _resolve('}')

        # Header key IDs — for sticky header-value state
        self._header_key_ids = set()
        for tok in ["M:", "L:", "K:", "V:", "Q:", "P:"]:
            tid = _resolve(tok)
            if tid is not None:
                self._header_key_ids.add(tid)

        # Per-batch tracking for header mode and chord-quote parity.
        # These are reset/grown as needed in __call__.
        # We use simple backward scans — the sequences are short enough
        # that this is cheaper than maintaining mutable per-batch state.

    def _is_inside_chord_quote(self, input_ids: torch.LongTensor) -> torch.BoolTensor:
        """Return a bool mask [batch] that is True where the sequence has an
        unmatched opening " (i.e. we are inside a chord symbol)."""
        if self._quote_id is None:
            return torch.zeros(input_ids.size(0), dtype=torch.bool,
                               device=input_ids.device)
        # Count " tokens in the full sequence; odd count = inside chord
        quote_counts = (input_ids == self._quote_id).sum(dim=1)
        return (quote_counts % 2) == 1

    def _is_in_header_line(self, input_ids: torch.LongTensor) -> torch.BoolTensor:
        """Return True where the most recent newline-or-BOS was followed by
        a header key token (M:, K:, etc.), meaning we're still on a header line."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        result = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if seq_len == 0:
            return result

        # Scan backward from the end to find the most recent newline (or start)
        # then check if the token right after it is a header key
        for b in range(batch_size):
            # Walk backward to find \n or beginning
            header_pos = -1
            for i in range(seq_len - 1, -1, -1):
                tid = input_ids[b, i].item()
                if self._newline_id is not None and tid == self._newline_id:
                    # Found newline — check the token right after it
                    if i + 1 < seq_len:
                        header_pos = i + 1
                    break
            else:
                # No newline found — check from position 0
                header_pos = 0

            if header_pos >= 0 and header_pos < seq_len:
                tid = input_ids[b, header_pos].item()
                if tid in self._header_key_ids:
                    result[b] = True

        return result

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = scores.size(0)
        device = scores.device

        if input_ids.shape[1] == 0:
            last_tokens = torch.full((batch_size,), self.bos_token_id,
                                     dtype=torch.long, device=device)
        else:
            last_tokens = input_ids[:, -1]

        current_states = self.token_to_state[last_tokens].clone()

        # ── Dynamic state adjustments ─────────────────────────────────

        # 1. Header-value sticky state: if we're on a header line (between
        #    a header key and the next newline), force state 15 so the model
        #    can only produce header-legal tokens (values, digits, pitches as
        #    key names, newline to exit). This prevents hallucinations like
        #    V:1{g or M:4/4!trill!
        if input_ids.shape[1] > 0:
            in_header = self._is_in_header_line(input_ids)
            # Don't override if the last token WAS the newline that exits
            is_newline = (last_tokens == self._newline_id) if self._newline_id is not None \
                         else torch.zeros(batch_size, dtype=torch.bool, device=device)
            sticky_mask = in_header & ~is_newline
            current_states[sticky_mask] = 15

        # 2. Chord quote toggle: if last token is " and we're inside an open
        #    chord (odd number of " tokens seen), this " is CLOSING → state 14.
        #    This handles "C", "Am7", "F#dim" etc. regardless of the primary
        #    state of the tokens between the quotes.
        if self._quote_id is not None:
            quote_mask = (last_tokens == self._quote_id)
            if quote_mask.any() and input_ids.shape[1] > 0:
                # After emitting this ", is the total count even? (was odd before,
                # this " closed it.) Count in the FULL sequence including this token.
                total_quotes = (input_ids == self._quote_id).sum(dim=1)
                # Even count = just closed a chord; odd = just opened one
                closing = quote_mask & (total_quotes % 2 == 0)
                current_states[closing] = 14

        # 3. Bracket ] close: if we're inside a bracket chord, ] closes it.
        #    Check previous token's state to confirm we were inside a bracket.
        if self._bracket_close_id is not None and input_ids.shape[1] > 1:
            bracket_close_mask = (last_tokens == self._bracket_close_id)
            prev_states = self.token_to_state[input_ids[:, -2]]
            closing_bracket = bracket_close_mask & (
                prev_states.eq(9) | prev_states.eq(10) | prev_states.eq(11) |
                prev_states.eq(12) | prev_states.eq(16)
            )
            current_states[closing_bracket] = 10  # treat like after-pitch

        # ── Apply mask ────────────────────────────────────────────────
        valid_mask = self.state_to_allowed[current_states]

        # Handle vocab size mismatch
        vs_logits = scores.size(-1)
        vs_mask = valid_mask.size(-1)
        if vs_mask > vs_logits:
            valid_mask = valid_mask[:, :vs_logits]
        elif vs_mask < vs_logits:
            pad = torch.ones((batch_size, vs_logits - vs_mask),
                             dtype=torch.bool, device=device)
            valid_mask = torch.cat([valid_mask, pad], dim=-1)

        scores[~valid_mask] = -float('inf')
        return scores


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience builder
# ═══════════════════════════════════════════════════════════════════════════

def build_abc_logits_processor(tokenizer, device="cpu"):
    """
    One-liner to build the grammar-constrained logits processor.

    Usage:
        from tokenizer import build_abc_tokenizer
        from static2_sync import build_abc_logits_processor

        tk = build_abc_tokenizer()
        processor = build_abc_logits_processor(tk, device="cuda")
        output = model.generate(..., logits_processor=[processor])
    """
    compiler = ABCGrammarCompiler(tokenizer)
    t2s, s2a, _ = compiler.build_state_tensors(device=device)
    return ABCLogitsProcessor(t2s, s2a, tokenizer)


# ═══════════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from tokenizer_sync import build_abc_tokenizer

    print("Building custom ABC tokenizer...")
    tk = build_abc_tokenizer()

    print("Compiling grammar state machine...")
    compiler = ABCGrammarCompiler(tk)
    t2s, s2a, token_cats = compiler.build_state_tensors(device="cpu")

    print(f"\n  Vocab size: {compiler.vocab_size}")
    print(f"  States: {s2a.shape[0]}")
    print(f"  Tokens with multiple categories: "
          f"{sum(1 for v in token_cats.values() if len(v) > 1)}")

    # ── Simulate sequences through the FULL logits processor ──────────
    processor = ABCLogitsProcessor(t2s, s2a, tk)
    vocab_size = len(tk)

    def resolve(tok_str):
        """Resolve a token string to its ID via encode()."""
        ids = tk.encode(tok_str, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def simulate(label, token_strs):
        """Run tokens through the actual logits processor and report."""
        print(f"\n  Simulating: {label}")
        # Build up input_ids incrementally, checking allowed at each step
        ids_so_far = []
        for tok in token_strs:
            tid = resolve(tok)
            if tid is None:
                print(f"    '{tok}' → NOT IN VOCAB (or not atomic)")
                return False

            # Create fake scores (all zeros)
            input_tensor = torch.tensor([ids_so_far], dtype=torch.long)
            fake_scores = torch.zeros(1, vocab_size)
            masked_scores = processor(input_tensor, fake_scores)

            is_allowed = masked_scores[0, tid].item() != float('-inf')
            # Get the state this token would produce
            state = t2s[tid].item()
            print(f"    '{tok}' (id={tid:3d}) → "
                  f"{'✓' if is_allowed else '✗ BLOCKED'}"
                  f"  (primary state → {state})")
            if not is_allowed:
                return False
            ids_so_far.append(tid)
        return True

    # Test 1: Header line — M:4/4 should stay in header mode
    ok1 = simulate("M:4/4\\n", ["M:", "4/4", "\n"])

    # Test 2: Voice line — V:1 clef=bass (space bounces through idle)
    ok2 = simulate("V:1 clef=bass\\n", ["V:", "1", " ", "clef=", "bass", "\n"])

    # Test 2b: Voice with transpose — V:1 transpose=-3
    ok2b = simulate("V:1 transpose=-3\\n",
                    ["V:", "1", " ", "transpose=", "-", "3", "\n"])

    # Test 3: Simple note with decoration
    ok3 = simulate("!mf! ^c'2", ["!mf!", "^", "c", "'", "2"])

    # Test 4: Chord quote with pitch root — "C" (the bug we fixed)
    ok4 = simulate('"C" c2', ['"', "C", '"', "c", "2"])

    # Test 5: Multi-token chord — "Am7"
    ok5 = simulate('"Am7" d2', ['"', "A", "m", "7", '"', "d", "2"])

    # Test 6: Bracket chord — [CEG]
    ok6 = simulate("[CEG]2", ["[", "C", "E", "G", "]", "2"])

    # Test 7: Grace note — {gag}c2
    ok7 = simulate("{gag}c2", ["{", "g", "a", "g", "}", "c", "2"])

    # Test 8: Full sequence per ABC spec
    ok8 = simulate("{g}\"C\"!mf!^c'2-",
                   ["{", "g", "}", '"', "C", '"', "!mf!", "^", "c", "'", "2", "-"])

    # Test 9: Header should NOT allow grace/decoration
    print("\n  Verifying: M: should NOT allow { or !trill!")
    m_id = resolve("M:")
    grace_id = resolve("{")
    trill_id = resolve("!trill!")
    input_m = torch.tensor([[m_id]], dtype=torch.long)
    fake_scores = torch.zeros(1, vocab_size)
    masked = processor(input_m, fake_scores)
    grace_blocked = masked[0, grace_id].item() == float('-inf')
    trill_blocked = masked[0, trill_id].item() == float('-inf')
    print(f"    {{ after M:: {'✓ blocked' if grace_blocked else '✗ ALLOWED (bug!)'}")
    print(f"    !trill! after M:: {'✓ blocked' if trill_blocked else '✗ ALLOWED (bug!)'}")

    # Summary
    results = [ok1, ok2, ok2b, ok3, ok4, ok5, ok6, ok7, ok8, grace_blocked, trill_blocked]
    passed = sum(results)
    print(f"\n  {'='*50}")
    print(f"  Self-test: {passed}/{len(results)} passed")
    if all(results):
        print("  All checks passed.")
    else:
        print("  ⚠ Some checks failed — review output above.")
    print(f"  {'='*50}")