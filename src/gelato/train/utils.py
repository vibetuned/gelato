import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.trainer_utils import get_last_checkpoint
logger = logging.getLogger(__name__)


def get_tokenizer(model_name: str = "google/gemma-3-270m") -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Gemma has eos but no dedicated pad token; reuse eos so the collator can mask padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Enforce right padding for standard causal LM training
    tokenizer.padding_side = "right"
    return tokenizer

class GelatoDataCollator:
    """
    Collator for GelatoModel. Batches images and tokenizes the text sequences.
    """
    def __init__(self, tokenizer, max_seq_len=2048):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        
        # 1. Append the EOS token string directly to the raw text
        texts = [f"{item['text']}{self.tokenizer.eos_token}" for item in batch]
        
        # 2. Single-pass Fast Tokenization (Silences the warning!)
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=True, # Handles the BOS token automatically
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        
        # 3. The Vectorized Truncation Fix
        # Universal way to find the last valid token index (works for both left/right padding)
        last_token_indices = attention_mask.cumsum(dim=1).argmax(dim=1) 
        
        # Overwrite the last valid token with the EOS ID. 
        # If it wasn't truncated, this just overwrites the EOS with an EOS.
        # If it was truncated, this replaces the chopped-off note with the required EOS token.
        input_ids[torch.arange(input_ids.size(0)), last_token_indices] = self.tokenizer.eos_token_id
        
        # 4. Clone labels and mask padding with -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
            
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def load_checkpoint(
    model, checkpoint_path: str, device: torch.device, eval: bool = True
):
    """Resolve a checkpoint path and load weights into the model.

    Handles:
      - HF Trainer output dirs (finds the last checkpoint-NNNN subdirectory)
      - Direct paths to model.safetensors or pytorch_model.bin
      - Single-run directories containing weight files

    Args:
        eval: If True (default), sets the model to eval mode. Set to False for training.
    """
    from transformers.trainer_utils import get_last_checkpoint
    
    ckpt = Path(checkpoint_path)
    if ckpt.is_dir():
        last = get_last_checkpoint(str(ckpt))
        if last is not None:
            ckpt = Path(last)
            logger.info(f"  Using last checkpoint: {ckpt}")

    for name in ("model.safetensors", "pytorch_model.bin"):
        weights = (ckpt / name) if ckpt.is_dir() else ckpt
        if weights.exists():
            if weights.suffix == ".safetensors":
                from safetensors.torch import load_file
                state = load_file(weights, device=str(device))
            else:
                state = torch.load(weights, map_location=device, weights_only=False)
            
            # --- THE TIED WEIGHTS FIX ---
            # Gemma 3 shares the embedding and LM head weights. 
            # If the checkpoint omitted the head, we must copy it from the embeddings manually.
            lm_head_key = "text_model.lm_head.weight"
            embed_key = "text_model.model.embed_tokens.weight"
            
            if lm_head_key not in state and embed_key in state:
                logger.info("  Restoring tied LM head weights from embeddings...")
                state[lm_head_key] = state[embed_key]
            # ----------------------------

            # Use strict=False to safely ignore HF buffers like RoPE caches
            missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
            
            # Filter out the keys we expect to be missing (if any)
            real_missing = [k for k in missing_keys if k != lm_head_key]
            if real_missing:
                logger.warning(f"  Missing keys during load: {real_missing}")
                
            logger.info(f"  Loaded weights: {weights}")
            break
    else:
        logger.warning("No weights file found — using random weights.")

    model.to(device)
    if eval:
        model.eval()