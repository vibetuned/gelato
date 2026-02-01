from transformers import AutoTokenizer, PreTrainedTokenizer

# Special Tokens
TOK_START = "<B>"
TOK_END = "<E>"
TOK_IMG = "<I>" # Placeholder for image latents if needed, or prompt marker
TOK_LINE = "$"  # Logical line break / System break
TOK_TEXT = "<text>" # Anonymized text

# We will let the user get ID dynamically, but providing a default placeholder if needed.
# For now, let's just make sure imports work. 
# IMG_TOKEN_ID meant to be the ID of TOK_IMG. 
# But we can't know ID until tokenizer is loaded.
# So we should rely on tokenizer.convert_tokens_to_ids(TOK_IMG).


ADDITIONAL_SPECIAL_TOKENS = [TOK_START, TOK_END, TOK_IMG, TOK_LINE, TOK_TEXT]

def get_tokenizer(model_name: str = "google/gemma-3-270m") -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens
    # Check if they exist to avoid duplication
    new_tokens = [t for t in ADDITIONAL_SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        
    return tokenizer

def resize_model_embeddings(model, tokenizer):
    model.resize_token_embeddings(len(tokenizer))
