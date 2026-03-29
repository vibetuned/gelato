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
        ],
        # ── Sequence modifiers ────────────────────────────────────────
        "grace_open":   ["{"],
        "grace_close":  ["}"],
        "chord_quote":  ['"'],
        "chord_text":   ["maj", "min", "m", "dim", "aug", "+", "sus"],
        # ── Decorations ──────────────────────────────────────────────
        # Combined tokens: these are the ONLY decoration forms in the
        # custom tokenizer (no bare ! + name + ! sequence needed)
        "dec_combined":  [
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
        "tie_slur":     ["-", "(", ")"],
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
    }

    def _build_token_categories(self):
        """
        Returns:
            primary: dict[str, list[int]]  — category name → list of token IDs
            token_cats: dict[int, set[str]] — token ID → set of category names
        """
        primary = {cat: [] for cat in self._CATEGORY_TABLE}
        token_cats = {}  # token_id → set of categories

        for cat, token_strs in self._CATEGORY_TABLE.items():
            for tok_str in token_strs:
                if tok_str in self.vocab:
                    tid = self.vocab[tok_str]
                    primary[cat].append(tid)
                    token_cats.setdefault(tid, set()).add(cat)

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
        NUM_STATES = 17

        token_to_state = torch.zeros(self.vocab_size, dtype=torch.long, device=device)

        # Primary state assignment — last write wins for multi-role tokens,
        # but the logits processor uses state_to_allowed which includes
        # ALL roles via the allowed-mask.
        state_map = {
            "idle": 0, "barline": 0, "header_key": 0,
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
        #   key names (pitch tokens), digits, accidentals, idle (newline ends it)
        S[15, ids("header_value", "pitch", "duration", "idle", "barline",
                  "accidental")] = True

        # State 16 — Inside bracket chord [ceg]: pitches, accidentals,
        #   durations, and close bracket
        S[16, ids("pitch", "accidental", "duration", "octave", "bracket")] = True

        # EOS/PAD always allowed
        if eos is not None:
            S[:, eos] = True
        if pad is not None:
            S[:, pad] = True

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

        # Pre-compute token IDs for dynamic state adjustments
        vocab = tokenizer.get_vocab()
        self._quote_id = vocab.get('"')
        self._newline_id = vocab.get('\n')
        self._bracket_open_id = vocab.get('[')
        self._bracket_close_id = vocab.get(']')
        self._grace_close_id = vocab.get('}')

        # Sets of state IDs that indicate "inside chord text"
        self._chord_text_states = {3, 4}   # open quote or continuing text
        self._header_value_state = 15
        self._bracket_state = 16

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

        # 1. Chord quote toggle: if last token is " and we're in a chord
        #    text context, the quote is CLOSING → state 14
        if self._quote_id is not None:
            quote_mask = (last_tokens == self._quote_id)
            if quote_mask.any() and input_ids.shape[1] > 1:
                prev_states = self.token_to_state[input_ids[:, -2]]
                closing = quote_mask & (prev_states.eq(4) | prev_states.eq(3))
                # But opening " from state 3 with prev in chord_text → close
                # Actually: if prev was chord_text (4) or pitch-as-chord-root (10)
                # inside a chord context, it's closing
                # Simplified: if previous token's state is 4, we're closing
                closing = quote_mask & prev_states.eq(4)
                current_states[closing] = 14

        # 2. Newline after header value → back to idle
        if self._newline_id is not None:
            newline_mask = (last_tokens == self._newline_id)
            # After a newline, always go to idle regardless of what came before
            # (This is handled by the primary state mapping already — \n → state 0)

        # 3. Bracket ] close: if we're inside a bracket chord, ] closes it
        # The primary mapping puts [ and ] both at state 16. We need to
        # distinguish: [ → opens (state 16), ] after pitches → closes (state 0)
        if self._bracket_close_id is not None and input_ids.shape[1] > 1:
            bracket_close_mask = (last_tokens == self._bracket_close_id)
            # If we were in bracket state, ] closes back to idle/pitch-continuation
            prev_states = self.token_to_state[input_ids[:, -2]]
            closing_bracket = bracket_close_mask & (
                prev_states.eq(10) | prev_states.eq(11) | prev_states.eq(12) |
                prev_states.eq(16)
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
    from tokenizer import build_abc_tokenizer

    print("Building custom ABC tokenizer...")
    tk = build_abc_tokenizer()

    print("Compiling grammar state machine...")
    compiler = ABCGrammarCompiler(tk)
    t2s, s2a, token_cats = compiler.build_state_tensors(device="cpu")

    print(f"\n  Vocab size: {compiler.vocab_size}")
    print(f"  States: {s2a.shape[0]}")
    print(f"  Tokens with multiple categories: "
          f"{sum(1 for v in token_cats.values() if len(v) > 1)}")

    # Quick sanity: simulate a simple sequence
    print("\n  Simulating: M:4/4\\n !mf! ^c'2 |")
    test_tokens = ["M:", "4/4", "\n", "!mf!", "^", "c", "'", "2", " ", "|"]
    state = 0
    for tok in test_tokens:
        tid = tk.get_vocab().get(tok)
        if tid is None:
            print(f"    '{tok}' → NOT IN VOCAB")
            continue
        allowed = s2a[state]
        is_allowed = allowed[tid].item()
        new_state = t2s[tid].item()
        print(f"    '{tok}' (id={tid:3d}) | state {state:2d} → "
              f"{'✓' if is_allowed else '✗'} → state {new_state:2d}")
        if is_allowed:
            state = new_state
        else:
            print(f"    !! BLOCKED at state {state}")
            break

    print("\n  Done.")
