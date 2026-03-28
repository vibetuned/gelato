import torch
import re
from transformers import AutoTokenizer, LogitsProcessor

class STATICGrammarCompiler:
    def __init__(self, tokenizer_name="google/gemma-3-270m"):
        print("Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vocab_size = len(self.tokenizer)
        self.vocab = self.tokenizer.get_vocab()

    def _categorize_vocabulary(self):
        categories = {
            "header":      [],  # L:, M:, K:, V:, w:, W:, Q:, m:
            "pitch":       [],  # Notes and rests: [=^_]?[A-Ga-gz][,']*
            "duration":    [],  # 1-9, /2, 3/4, etc.
            "barline":     [],  # |  |:  :|  |]  ||
            "whitespace":  [],  # Spaces and tabs only (NOT newlines)
            "linebreak":   [],  # \n tokens
            "chord_open":  [],  # [
            "chord_close": [],  # ]
            "text":        [],  # Lyrics, annotations, markup, everything else
        }

        for token_str, token_id in self.vocab.items():
            clean_str = self.tokenizer.decode([token_id]).strip()
            raw_str = self.tokenizer.decode([token_id])  # Unstripped for whitespace detection

            # Header fields that survive the stripped format
            if (re.fullmatch(r"^[LMKVXTPCwqQm]:.*", clean_str) 
                or clean_str in ["L", "M", "K", "V", "X", "T", "P", "w", "W", "Q", "m", ":", "bass", "treble", "clef", "none"]
                or re.fullmatch(r"^[A-G][b#]$", clean_str)): # K:Bb, K:F#
                categories["header"].append(token_id)

            elif "\n" in raw_str:
                categories["linebreak"].append(token_id)

            elif re.fullmatch(r"^[ \t]+$", raw_str):
                categories["whitespace"].append(token_id)

            elif re.fullmatch(r"^[=^_]*[A-Ga-gz]+[,']*$", clean_str) or re.fullmatch(r"^[,']+\/?\d*$", clean_str):
                categories["pitch"].append(token_id)

            elif clean_str and re.fullmatch(r"^[1-9]*\/?[1-9]*$", clean_str):
                categories["duration"].append(token_id)
                
            elif re.fullmatch(r"^\(\d*", clean_str): # Tuplets like (3
                categories["duration"].append(token_id)

            elif re.fullmatch(r"^\|[\]:|]?|:\|$", clean_str):
                categories["barline"].append(token_id)

            elif re.fullmatch(r"^\[$", clean_str):
                categories["chord_open"].append(token_id)

            elif re.fullmatch(r"^[,']*\]$", clean_str):
                categories["chord_close"].append(token_id)
            
            else:
                categories["text"].append(token_id)

        return categories

    def build_state_tensors(self, device="cuda"):
        categories = self._categorize_vocabulary()

        # State mapping:
        # 0: Fallback     (allows everything — safety net)
        # 1: Header       (after L:, M:, K:, V: tokens)
        # 2: Pitch        (after a note or rest)
        # 3: Duration     (after a number)
        # 4: Barline      (after |, |:, etc.)
        # 5: Whitespace   (after a space)
        # 6: Linebreak    (after \n)
        # 7: Chord open   (after [)
        # 8: Chord close  (after ])
        # 9: Text         (lyrics/annotations)
        num_states = 10

        token_to_state = torch.zeros(self.vocab_size, dtype=torch.long, device=device)
        token_to_state[categories["header"]]      = 1
        token_to_state[categories["pitch"]]        = 2
        token_to_state[categories["duration"]]     = 3
        token_to_state[categories["barline"]]      = 4
        token_to_state[categories["whitespace"]]   = 5
        token_to_state[categories["linebreak"]]    = 6
        token_to_state[categories["chord_open"]]   = 7
        token_to_state[categories["chord_close"]]  = 8
        token_to_state[categories["text"]]         = 9

        state_to_allowed = torch.zeros(
            (num_states, self.vocab_size), dtype=torch.bool, device=device
        )

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        eos_list = [eos_id] if eos_id is not None else []

        # Shorthand
        h = categories["header"]
        p = categories["pitch"]
        d = categories["duration"]
        b = categories["barline"]
        w = categories["whitespace"]
        lb = categories["linebreak"]
        co = categories["chord_open"]
        cc = categories["chord_close"]
        t = categories["text"]

        # State 0 — Fallback: allow everything
        state_to_allowed[0, :] = True

        # State 1 — Header: header tokens can continue (K:C has multiple tokens),
        #   or a linebreak ends the header field, or another header starts
        state_to_allowed[1, h + w + lb + d + p + t + eos_list] = True

        # State 2 — After pitch: expect duration, another pitch (in chords),
        #   barline, whitespace, linebreak, chord close, or EOS
        state_to_allowed[2, d + p + b + w + lb + cc + t + eos_list] = True

        # State 3 — After duration: the note is complete, expect next pitch,
        #   barline, whitespace, linebreak, chord open, or EOS
        state_to_allowed[3, p + b + w + lb + co + t + eos_list] = True

        # State 4 — After barline: expect pitch, whitespace, linebreak,
        #   chord open, or EOS (end of piece)
        state_to_allowed[4, p + w + lb + co + t + eos_list] = True

        # State 5 — After whitespace: expect pitch, barline, duration,
        #   linebreak, chord open, or EOS
        state_to_allowed[5, p + b + d + lb + co + t + eos_list] = True

        # State 6 — After linebreak: this is where a new line starts.
        #   Could be a header field, pitch, barline, chord open, 
        #   another linebreak, or EOS
        state_to_allowed[6, h + p + b + co + lb + t + eos_list] = True

        # State 7 — After chord open [: ONLY pitches allowed inside chords
        state_to_allowed[7, p + t] = True

        # State 8 — After chord close ]: expect duration (the chord's rhythm)
        state_to_allowed[8, d + w + b + lb + t + eos_list] = True

        # State 9 — After text: allow everything
        state_to_allowed[9, t + w + lb + p + d + b + co + cc + h + eos_list] = True

        # Ensure EOS/PAD always self-loop for clean termination
        if eos_id is not None:
            state_to_allowed[:, eos_id] = True
        if pad_id is not None:
            state_to_allowed[:, pad_id] = True

        return token_to_state, state_to_allowed


class STATICLogitsProcessor(LogitsProcessor):
    def __init__(self, token_to_state: torch.Tensor, state_to_allowed: torch.Tensor):
        self.token_to_state = token_to_state
        self.state_to_allowed = state_to_allowed
        self.bos_token_id = 2 # Standard Gemma <bos> token ID

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Get the very last token generated for each sequence in the batch
        if input_ids.shape[1] == 0:
            last_tokens = torch.full((scores.size(0),), self.bos_token_id, dtype=torch.long, device=scores.device)
        else:
            last_tokens = input_ids[:, -1]
            
        # 1. Fully Vectorized Lookup: Get the current state for the whole batch simultaneously
        current_states = self.token_to_state[last_tokens]
        
        # 2. Extract the boolean mask of allowed next tokens for those states
        valid_mask = self.state_to_allowed[current_states]

        vocab_size_logits = scores.size(-1)
        vocab_size_mask = valid_mask.size(-1)
        
        if vocab_size_mask > vocab_size_logits:
            # Truncate the extra tokens from the mask (tokenizer has extra tokens not in lm_head)
            valid_mask = valid_mask[:, :vocab_size_logits]
        elif vocab_size_mask < vocab_size_logits:
            # Pad the mask with True to safely allow dummy tokens the model might output
            padding = torch.ones(
                (valid_mask.size(0), vocab_size_logits - vocab_size_mask), 
                dtype=torch.bool, 
                device=valid_mask.device
            )
            valid_mask = torch.cat([valid_mask, padding], dim=-1)
        
        # 3. Apply the mask
        scores[~valid_mask] = -float('Inf')
        
        return scores