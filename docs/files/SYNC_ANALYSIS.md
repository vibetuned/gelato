# Synchronization Analysis: tokenizer.py ↔ static2.py ↔ strip_abc.py

## CRITICAL: Tokenizer Mismatch (static2.py)

**`static2.py` uses `google/gemma-3-270m`** — a general-purpose LLM tokenizer with ~256k subword tokens.  
**`tokenizer.py` builds a custom ABC tokenizer** with ~150 domain-specific tokens.

These are completely different vocabularies. The logits processor's regex-based `_categorize_vocabulary()` was designed to reverse-engineer ABC meaning from Gemma's subword pieces, but your custom tokenizer makes that unnecessary — every token already *is* an ABC concept. The entire classification layer should be replaced with a deterministic lookup table.

---

## Bug-by-bug Breakdown

### 1. tokenizer.py Issues

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| T1 | **Ambiguous pitch/key tokens**: `C G D F A E B` appear once in vocab but serve as both key signatures (after `K:`) and pitches (in note lines) | Logits processor can't distinguish context from token ID alone | Keep single token; handle in state machine (header state allows these after `K:`) |
| T2 | **Ambiguous short-dec/header tokens**: `M P` are both short decorations AND the start of `M:` `P:` headers — but `M:` and `P:` are separate tokens, so `M` alone is fine as a decoration | Low — greedy matching picks `M:` for headers | None needed — verify greedy match works |
| T3 | **No tuplet token**: `(3` (triplet), `(p:q:r` tuplet notation is missing | Model can't produce tuplets | Add `(3` or handle `(` + digit sequences |
| T4 | **`[` and `]` are separate tokens** but `[ceg]` chord notation needs them as brackets around notes | Works if state machine allows `[` → pitch sequences → `]` | Add chord-bracket states to logits processor |
| T5 | **Chord symbol `"Am7"` tokenizes as `"` `A` `m` `7` `"`** — 5 tokens | State machine must allow multi-token chord text loops | Fix State 4 to loop on chord_text tokens |
| T6 | **Missing key modifiers**: `Dor`, `Mix`, `Phr`, `Lyd`, `Loc`, `min` as key modes after `K:` | Can't express modal keys | Add mode tokens or handle in header state |
| T7 | **`#` not in vocab** but appears in chord symbols like `F#7` | Would produce `<unk>` inside chords | Add `#` as a token |
| T8 | **Grace note `{/g}` needs `/` inside braces** — state machine doesn't allow `/` after `{` | Can't produce acciaccatura grace notes | Allow `/` in grace state |
| T9 | **`add_special_tokens` on a no-merges BPE** — the tokenizer has no base character vocabulary and no merges, so any input character not matching a special token maps to `<unk>` | Characters like `<`, `>`, `#`, `:` in isolation produce `<unk>` | This is actually *by design* for a constrained tokenizer — just ensure ALL needed characters/tokens are in the vocab |

### 2. static2.py Issues

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| S1 | **Wrong tokenizer** — uses Gemma, not the custom ABC tokenizer | Every category assignment is wrong | Switch to custom tokenizer, use lookup table |
| S2 | **`chord_text` regex `^[A-Ga-g0-9#bmsudj\+]+$`** matches way too many Gemma tokens (any token containing only those chars) | Massive false-positive pollution of chord_text category | With custom tokenizer, explicitly list chord tokens |
| S3 | **State 4 (chord text) → State 3 (close quote) only** — but chord names span multiple tokens (`A` + `m` + `7`) | After first chord token, only `"` is allowed — can't complete "Am7" | State 4 must also allow more chord_text tokens (self-loop) |
| S4 | **State 1 (grace open) allows only one accid+pitch** — but grace notes contain multiple notes `{gag}` | Only single-note grace notes work | State 1 needs pitch → pitch looping + `}` exit |
| S5 | **No header-value state** — after `M:` the machine is in state 0 (idle) which tries to interpret `4/4` as note sequence | Header values get blocked or misinterpreted | Add dedicated header states |
| S6 | **Dec close `!` maps to state 5** (dec_bound) which expects a dec_name — but closing `!` should transition forward | `!trill!` works: `!`→state5→`trill`→state6→`!`→state5... infinite loop | Distinguish opening vs closing `!` (like the chord-quote toggle) |
| S7 | **`bos_token_id = 2` hardcoded** | Wrong if using a different tokenizer | Read from tokenizer: `tokenizer.bos_token_id` |
| S8 | **No `[ceg]` chord bracket handling** | Inline chords impossible | Add bracket states |
| S9 | **Tuplet `(3` not handled** | Blocked as invalid | Add tuplet state or treat `(` as context-dependent |

### 3. strip_abc.py Issues

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| R1 | **`_INLINE_COMMENT` regex `\s*%.*$`** strips from first `%` — but `%%` directives are handled separately and `%abc-2.1` header should be stripped too | Could strip valid content if `%` appears in chord annotations (unlikely but possible) | Minor — acceptable for training data |
| R2 | **Keeps `V:` voice attributes** like `clef=bass` — but `clef=bass` isn't in the tokenizer vocab as a unit | Tokenizer would need `clef=bass` or the stripper should further simplify voice lines | Either add `clef=` tokens or strip to just `V:1` |
| R3 | **No stripping of `w:` lyrics lines** | `w:` matches `_KEEP_HEADERS`? No — `_KEEP_HEADERS` is `^[MLKVQP]:` — lowercase `w:` would match `_ANY_HEADER` but NOT `_KEEP_HEADERS`, so it IS stripped. Good. | None |
| R4 | **Output has no trailing context** — stripped files end with `\n` which is clean | None | None |

### 4. Cross-Script Sync Issues

| # | Issue | Scripts |
|---|-------|---------|
| X1 | strip_abc keeps `V:1 clef=bass` but tokenizer has no `clef=` token | strip ↔ tokenizer |
| X2 | strip_abc output may contain `(3` tuplets but tokenizer has no tuplet token | strip ↔ tokenizer |
| X3 | Logits processor doesn't know about header lines at all — it would try to constrain `4/4` after `M:` as a note sequence | tokenizer ↔ static2 |
| X4 | The `"` chord quote toggle logic assumes exactly `"` + one chord_text + `"` — but the tokenizer splits chord names into multiple tokens | tokenizer ↔ static2 |
| X5 | Grace notes in stripped ABC like `{/gag}` need `/`, multiple pitches, and `}` — state machine only allows one pitch | strip ↔ static2 |

---

## Recommended Fix Priority

1. **S1+S2**: Rewrite `static2.py` to use the custom tokenizer with a deterministic lookup table
2. **S3+S4+S5+S6**: Fix the state machine transitions (chord loop, grace loop, header state, dec-boundary toggle)
3. **T3+T6+T7**: Add missing tokens to the tokenizer (tuplet, modes, `#`)
4. **X1**: Decide on voice-line handling (simplify in stripper or add tokens)
5. **T5+X4**: Already works with state machine fix S3
