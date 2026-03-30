import os
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from gelato.model.layers import EngramConfig, NgramHashMapping, MultiHeadEmbedding

def extract_and_warm_start(
    abc_dir="data/dataset-small/abcs", 
    model_name="google/gemma-3-270m",
    custom_tokenizer_path=None,
    top_k_ngrams=10000,
    output_path="checkpoints/engram_warm_start.pt"
):
    print(f"Loading Tokenizer and Base Model: {model_name}")
    old_tokenizer = AutoTokenizer.from_pretrained(model_name)
    # We load the base model just to steal its embedding weights
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if custom_tokenizer_path and os.path.exists(custom_tokenizer_path):
        print(f"Loading Custom Tokenizer from: {custom_tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(custom_tokenizer_path)
    else:
        print("Building Custom ABC Tokenizer dynamically...")
        from gelato.model.tokenizer import build_abc_tokenizer
        tokenizer = build_abc_tokenizer(save_dir=None)

    print("Resizing and inheriting embeddings from old tokenizer...")
    old_embeddings = base_model.get_input_embeddings().weight.data.clone()
    base_model.resize_token_embeddings(len(tokenizer))
    new_embeddings = base_model.get_input_embeddings().weight.data
    mean_emb = old_embeddings.mean(dim=0)
    
    with torch.no_grad():
        new_embeddings.copy_(mean_emb.unsqueeze(0).expand_as(new_embeddings))
    
    for new_id in range(len(tokenizer)):
        token_str = tokenizer.convert_ids_to_tokens(new_id)
        decoded_str = tokenizer.decode([new_id], skip_special_tokens=False)
        old_ids = old_tokenizer.encode(decoded_str, add_special_tokens=False)
        if len(old_ids) != 1:
            old_ids_str = old_tokenizer.encode(token_str, add_special_tokens=False)
            if len(old_ids_str) == 1:
                old_ids = old_ids_str
        if len(old_ids) == 1:
            with torch.no_grad():
                new_embeddings[new_id] = old_embeddings[old_ids[0]]
                
    # Save the base model and tokenizer NOW so EngramConfig can load it natively
    hf_save_path = os.path.join(os.path.dirname(output_path), "gemma-3-gelato-resized")
    os.makedirs(hf_save_path, exist_ok=True)
    print(f"Saving surgically resized HuggingFace model and tokenizer to {hf_save_path}...")
    base_model.save_pretrained(hf_save_path)
    tokenizer.save_pretrained(hf_save_path)
    
    # Engram mapping will use the saved native directory
    engram_cfg = EngramConfig(tokenizer_name_or_path=hf_save_path)

    base_embeddings = base_model.get_input_embeddings().weight.detach()
    
    # Initialize the Engram Hash Mapping to know WHERE to put the weights
    hash_mapping = NgramHashMapping(
        engram_vocab_size=engram_cfg.engram_vocab_size,
        max_ngram_size=engram_cfg.max_ngram_size,
        n_embed_per_ngram=engram_cfg.n_embed_per_ngram,
        n_head_per_ngram=engram_cfg.n_head_per_ngram,
        layer_ids=engram_cfg.layer_ids,
        tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
        pad_id=engram_cfg.pad_id,
        seed=engram_cfg.seed,
    )
    
    # Initialize the empty Engram memory table
    list_of_N = [x for y in hash_mapping.vocab_size_across_layers[engram_cfg.layer_ids[0]] for x in y]
    D = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram
    engram_memory = MultiHeadEmbedding(list_of_N=list_of_N, D=D)
    
    # We will initialize the table with near-zero noise first
    nn.init.normal_(engram_memory.embedding.weight, mean=0.0, std=0.02)

    print(f"Scanning ABC files in {abc_dir}...")
    abc_files = list(Path(abc_dir).glob("*.abc"))
    
    ngram_counts = Counter()
    
    # 1. Extract and Count N-grams
    for file_path in tqdm(abc_files, desc="Tokenizing & Counting"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            
        # Tokenize the whole file
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        
        # Sliding window for 2-grams, 3-grams, and 4-grams
        for n in range(2, engram_cfg.max_ngram_size + 1):
            for i in range(len(token_ids) - n + 1):
                ngram = tuple(token_ids[i : i + n])
                ngram_counts[ngram] += 1

    print(f"Found {len(ngram_counts)} unique N-grams. Slicing top {top_k_ngrams}...")
    most_common_ngrams = ngram_counts.most_common(top_k_ngrams)

    # 2. Inject Gemma's Knowledge into the Hash Table
    print("Warm-starting Engram memory slots...")
    layer_id = engram_cfg.layer_ids[0]
    
    with torch.no_grad():
        for ngram, count in tqdm(most_common_ngrams, desc="Injecting Embeddings"):
            ngram_tensor = torch.tensor([list(ngram)], dtype=torch.long)
            
            # Find exactly which slots this sequence hashes to
            # hash() returns [Batch, Seq_Len, Heads]
            hash_indices = hash_mapping.hash(ngram_tensor.numpy())[layer_id][0, -1, :] 
            
            # Get Gemma's base embeddings for these specific tokens
            # Shape: [N, Hidden_Dim]
            gemma_vectors = base_embeddings[list(ngram)]
            
            # Pool the sequence into a single representation (Mean Pooling)
            pooled_vector = gemma_vectors.mean(dim=0) 
            
            # We need to project Gemma's 1024-dim vector down to Engram's Head Dimension (D)
            # A simple slicing or linear downsample works for initialization
            downsampled_vector = pooled_vector[:D] 
            
            # Inject this learned representation into every hash head assigned to this N-gram
            for head_idx, hash_val in enumerate(hash_indices):
                # Calculate absolute index in the flattened embedding table
                absolute_idx = hash_val + engram_memory.offsets[head_idx]
                engram_memory.embedding.weight[absolute_idx] = downsampled_vector

    # 3. Save the warm-started weights
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(engram_memory.state_dict(), output_path)
    print(f"✅ Warm-start Engram weights saved to {output_path}")
    print(f"✅ Base model successfully saved! Update your YAML config:")
    print(f"   text_model_name: '{hf_save_path}'")

def main():
    parser = argparse.ArgumentParser(description="Extract and warm-start Engram embeddings.")
    parser.add_argument("--abc-dir", "--abc_dir", dest="abc_dir", type=str, default="data/dataset-small/abcs", help="Directory containing ABC files.")
    parser.add_argument("--model-name", "--model_name", dest="model_name", type=str, default="google/gemma-3-270m", help="Base model name.")
    parser.add_argument("--custom-tokenizer-path", "--custom_tokenizer_path", dest="custom_tokenizer_path", type=str, default=None, help="Path to custom tokenizer (if omitted, will build dynamically).")
    parser.add_argument("--top-k-ngrams", "--top_k_ngrams", dest="top_k_ngrams", type=int, default=10000, help="Number of Top-K n-grams to warm start.")
    parser.add_argument("--output-path", "--output_path", dest="output_path", type=str, default="checkpoints/pretrain/engram_warm_start.pt", help="Path to save weights.")
    
    args = parser.parse_args()
    
    extract_and_warm_start(
        abc_dir=args.abc_dir,
        model_name=args.model_name,
        custom_tokenizer_path=args.custom_tokenizer_path,
        top_k_ngrams=args.top_k_ngrams,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()