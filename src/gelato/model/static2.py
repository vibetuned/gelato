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
            "idle":         [],  # Whitespace, newlines
            "barline":      [],  # |, ||, [1, etc.
            "header":       [],  # Standard A-Z: fields
            "grace_open":   [],  # {
            "grace_close":  [],  # }
            "chord_quote":  [],  # "
            "chord_text":   [],  # Am7, maj, dim, etc.
            "dec_bound":    [],  # !
            "dec_name":     [],  # trill, fermata, etc.
            "dec_combined": [],  # !trill! (if tokenized as one)
            "dec_short":    [],  # ., ~, H, L, M, P, S, T, u, v
            "accid":        [],  # ^, _, =, ^^, __
            "pitch":        [],  # A-G, a-g, z, Z, x, y
            "octave":       [],  # ,, '
            "duration":     [],  # 1-9, /
            "tie_slur":     [],  # -, (, )
        }

        # Strict exact-match lists from the ABC reference guide
        dec_names = [
            "trill", "lowermordent", "uppermordent", "mordent", "pralltriller", 
            "accent", ">", "emphasis", "fermata", "invertedfermata", "tenuto", 
            "trem1", "trem2", "trem3", "trem4", "xstem", "slide", "turnx", 
            "invertedturnx", "arpeggio", "segno", "coda", "D.S.", "D.C.", 
            "dacoda", "dacapo", "fine", "shortphrase", "mediumphrase", 
            "longphrase", "upbow", "downbow", "thumb", "snap", "turn", 
            "roll", "breath", "plus", "wedge", "open", "crescendo(", "<(", 
            "crescendo)", "<)", "diminuendo(", ">(", "diminuendo)", ">)", 
            "pppp", "ppp", "pp", "p", "mp", "mf", "f", "ff", "fff", "ffff", "sfz"
        ] # [cite: 278-298, 307-311, 328-330, 333, 336, 351-355, 360, 362-370, 381, 383, 385-390, 397-398]

        dec_combined_exact = [f"!{name}!" for name in dec_names]

        for token_str, token_id in self.vocab.items():
            clean_str = self.tokenizer.decode([token_id]).strip()
            raw_str = self.tokenizer.decode([token_id])

            # 1. Idle & Structural
            if re.fullmatch(r"^[ \t\n]+$", raw_str):
                categories["idle"].append(token_id)
            elif re.fullmatch(r"^\|[\]:|]?|:\||\[[12]$", clean_str): # [cite: 260, 266-267]
                categories["barline"].append(token_id)
            elif re.fullmatch(r"^[A-Za-z]:.*", clean_str): # [cite: 5-9, 36, 41, 46, 52, 56, 58, 60-61, 63, 66, 68-69, 89-94, 96-97]
                categories["header"].append(token_id)

            # 2. Sequence Modifiers
            elif clean_str == "{": # [cite: 239]
                categories["grace_open"].append(token_id)
            elif clean_str == "}": # [cite: 239]
                categories["grace_close"].append(token_id)
            elif clean_str == '"': # [cite: 277]
                categories["chord_quote"].append(token_id)
            elif re.fullmatch(r"^[A-Ga-g0-9#bmsudj\+]+$", clean_str): # [cite: 276-291]
                categories["chord_text"].append(token_id)
            elif clean_str == "!": # [cite: 362]
                categories["dec_bound"].append(token_id)
            elif clean_str in dec_names:
                categories["dec_name"].append(token_id)
            elif clean_str in dec_combined_exact:
                categories["dec_combined"].append(token_id)
            elif clean_str in [".", "~", "H", "L", "M", "P", "S", "T", "u", "v"]: # [cite: 230, 334, 337, 339, 341, 344, 346, 348]
                categories["dec_short"].append(token_id)
            elif clean_str in ["^", "^^", "_", "__", "="]: # [cite: 223, 226, 228, 232, 234]
                categories["accid"].append(token_id)

            # 3. Core Note Elements
            elif re.fullmatch(r"^[A-Ga-g]$|^[zZxXy]$", clean_str): # [cite: 213-218, 238, 241, 245-246, 250]
                categories["pitch"].append(token_id)
            elif clean_str in [",", "'", ",,", "''"]: # [cite: 213, 217]
                categories["octave"].append(token_id)
            elif re.fullmatch(r"^[0-9]+$|^\/[0-9]*$", clean_str): # [cite: 53, 237, 243]
                categories["duration"].append(token_id)
            elif re.fullmatch(r"^[-()]+$", clean_str) or clean_str in [".(", ".)"]: # [cite: 224, 230, 235, 254]
                categories["tie_slur"].append(token_id)

        return categories

    def build_state_tensors(self, device="cuda"):
        categories = self._categorize_vocabulary()
        num_states = 15

        token_to_state = torch.zeros(self.vocab_size, dtype=torch.long, device=device)
        
        # Priority mapping (overwrites determine entry point)
        token_to_state[categories["idle"]]         = 0
        token_to_state[categories["barline"]]      = 0
        token_to_state[categories["header"]]       = 0
        token_to_state[categories["grace_open"]]   = 1
        token_to_state[categories["grace_close"]]  = 2
        token_to_state[categories["chord_quote"]]  = 3
        token_to_state[categories["chord_text"]]   = 4
        token_to_state[categories["dec_bound"]]    = 5
        token_to_state[categories["dec_name"]]     = 6
        token_to_state[categories["dec_combined"]] = 7
        token_to_state[categories["dec_short"]]    = 8
        token_to_state[categories["accid"]]        = 9
        token_to_state[categories["pitch"]]        = 10
        token_to_state[categories["octave"]]       = 11
        token_to_state[categories["duration"]]     = 12
        token_to_state[categories["tie_slur"]]     = 13

        # State 3 (Chord quote) toggles. If we enter from idle, it's open. 
        # But if we enter from chord_text, it's closing and maps to 14.
        
        state_to_allowed = torch.zeros((num_states, self.vocab_size), dtype=torch.bool, device=device)

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        eos_list = [eos_id] if eos_id is not None else []

        # Shortcuts
        idle = categories["idle"] + categories["barline"] + categories["header"] + eos_list
        g_op = categories["grace_open"]
        g_cl = categories["grace_close"]
        c_qt = categories["chord_quote"]
        c_tx = categories["chord_text"]
        d_bd = categories["dec_bound"]
        d_nm = categories["dec_name"]
        d_cb = categories["dec_combined"]
        d_sh = categories["dec_short"]
        acc = categories["accid"]
        pch = categories["pitch"]
        octv = categories["octave"]
        dur = categories["duration"]
        tsl = categories["tie_slur"]

        # START OF STRICT SEQUENCE: <grace><chord><decoration><accidental><note><length><tie> 

        # State 0 (Idle): Can start anywhere in the sequence down to Pitch
        state_to_allowed[0, g_op + c_qt + d_bd + d_cb + d_sh + acc + pch + idle] = True

        # State 1 (Inside Grace '{'): Simplify by jumping straight to pitch/accidental
        state_to_allowed[1, acc + pch] = True

        # State 2 (After Grace '}'): Must move forward to Chord, Dec, Accid, or Pitch
        state_to_allowed[2, c_qt + d_bd + d_cb + d_sh + acc + pch] = True

        # State 3 (Open Chord Quote '"'): Must output chord text
        state_to_allowed[3, c_tx] = True

        # State 4 (Inside Chord Text): Must close the quote
        state_to_allowed[4, c_qt] = True

        # State 14 (Close Chord Quote): Custom handler in logits processor will map closing quote here
        # Moves forward to Dec, Accid, or Pitch
        state_to_allowed[14, d_bd + d_cb + d_sh + acc + pch] = True

        # State 5 (Decoration Boundary '!'): Must output a valid decoration name
        state_to_allowed[5, d_nm] = True

        # State 6 (Inside Dec Name): Must close the boundary '!'
        state_to_allowed[6, d_bd] = True

        # State 7 (After Dec Close '!...!' or Combined): Moves forward to Accid or Pitch
        state_to_allowed[7, d_bd + d_cb + d_sh + acc + pch] = True

        # State 8 (After Dec Short): Moves forward to Accid or Pitch
        state_to_allowed[8, acc + pch] = True

        # State 9 (After Accidental): MUST output a Pitch
        state_to_allowed[9, pch] = True

        # State 10 (After Pitch): Can move to Octave, Duration, Tie, or end group
        state_to_allowed[10, octv + dur + tsl + idle] = True

        # State 11 (After Octave): Can move to Duration, Tie, or end group
        state_to_allowed[11, dur + tsl + idle] = True

        # State 12 (After Duration): Can move to Tie or end group
        state_to_allowed[12, tsl + idle] = True

        # State 13 (After Tie/Slur): Loop back to Idle or next sequence
        state_to_allowed[13, idle + g_op + c_qt + d_bd + d_cb + d_sh + acc + pch] = True

        if eos_id is not None:
            state_to_allowed[:, eos_id] = True
        if pad_id is not None:
            state_to_allowed[:, pad_id] = True

        return token_to_state, state_to_allowed


class STATICLogitsProcessor(LogitsProcessor):
    def __init__(self, token_to_state: torch.Tensor, state_to_allowed: torch.Tensor):
        self.token_to_state = token_to_state
        self.state_to_allowed = state_to_allowed
        self.bos_token_id = 2 

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] == 0:
            last_tokens = torch.full((scores.size(0),), self.bos_token_id, dtype=torch.long, device=scores.device)
            prev_tokens = last_tokens
        else:
            last_tokens = input_ids[:, -1]
            prev_tokens = input_ids[:, -2] if input_ids.shape[1] > 1 else last_tokens
            
        current_states = self.token_to_state[last_tokens].clone()
        prev_states = self.token_to_state[prev_tokens]
        
        # Dynamic State Adjustment: Handle closing chord quotes
        # If the last token was a quote (State 3), but the previous token was chord text (State 4), 
        # we are actually closing the chord, so transition to State 14
        closing_quote_mask = (current_states == 3) & (prev_states == 4)
        current_states[closing_quote_mask] = 14
        
        valid_mask = self.state_to_allowed[current_states]

        vocab_size_logits = scores.size(-1)
        vocab_size_mask = valid_mask.size(-1)
        
        if vocab_size_mask > vocab_size_logits:
            valid_mask = valid_mask[:, :vocab_size_logits]
        elif vocab_size_mask < vocab_size_logits:
            padding = torch.ones(
                (valid_mask.size(0), vocab_size_logits - vocab_size_mask), 
                dtype=torch.bool, 
                device=valid_mask.device
            )
            valid_mask = torch.cat([valid_mask, padding], dim=-1)
        
        scores[~valid_mask] = -float('Inf')
        
        return scores