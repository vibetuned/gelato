import argparse
import os
import json
import torch
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
from transformers import SiglipImageProcessor

from gelato.model.modeling import GelatoModel, GelatoConfig
from gelato.data.dataset import GelatoDataset, GelatoCollator
from gelato.utils.tokenizer import get_tokenizer, TOK_START, TOK_END
from gelato.metrics.errot_rates import compute_error_rates

def main():
    parser = argparse.ArgumentParser(description="Evaluate Gelato Model")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/full_run_1")
    parser.add_argument("--data_dir", type=str, default="./data/processed_test")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to evaluate for a quick test")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Initialize Tokenizer & Processor
    print("Initializing tokenizer and processor...")
    tokenizer = get_tokenizer()
    processor = SiglipImageProcessor.from_pretrained("google/siglip2-large-patch16-512")

    # 2. Initialize Model
    print("Loading model...")
    config = GelatoConfig()
    model = GelatoModel(config)
    model.resize_token_embeddings(len(tokenizer))
    
    # Load weights
    weights_path = os.path.join(args.checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        print(f"Warning: No weights found at {weights_path}, using untrained model!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 3. Load Dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = GelatoDataset(
        data_dir=Path(args.data_dir),
        tokenizer=tokenizer,
        split="test"
    )
    
    collator = GelatoCollator(processor, tokenizer)

    prompt_ids = tokenizer(TOK_START, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    # Ensure it's rank 1 or 2 as needed. Tokenizer usually returns [1, seq]
    
    predictions = []
    references = []
    
    limit = min(args.limit, len(dataset)) if args.limit > 0 else len(dataset)
    print(f"Evaluating {limit} samples...")

    for i in tqdm(range(limit)):
        try:
            sample = dataset[i]
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue
            
        # Manually collate a single sample to get pixel_values
        batch = collator([sample])
        pixel_values = batch["pixel_values"].to(device)
        
        # Ground truth
        ref_text = sample["abc_text"]
        
        # The true input ids used for reference
        ref_ids = batch["input_ids"][0]
        
        image_attention_mask = batch["image_attention_mask"].to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values=pixel_values,
                image_attention_mask=image_attention_mask,
                prompt_ids=prompt_ids,
                max_new_tokens=1024,
                do_sample=False, # greedy
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids(TOK_END)
            )
        
        # output_ids usually includes prompt_ids. But check how the model's text_model.generate works: 
        # it generates completion including prompt if input_embeds is used? Wait, if we use inputs_embeds,
        # generate returns the generated tokens starting from what was generated, or maybe it returns 
        # nothing but the first token if we passed embeds? Actually generate with inputs_embeds usually 
        # only returns the generated tokens!
        
        pred_ids = output_ids[0].cpu().tolist()
        
        predictions.append(pred_ids)
        references.append(ref_ids.cpu().tolist())
        
        # Save visual comparison for the first few
        if i % 5 == 0:
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            
            sample_id = sample.get("id", f"sample_{i}")
            pred_abc_path = os.path.join(args.output_dir, f"{sample_id}_pred.abc")
            ref_abc_path = os.path.join(args.output_dir, f"{sample_id}_ref.abc")
            
            with open(pred_abc_path, "w") as f:
                f.write(pred_text)
            with open(ref_abc_path, "w") as f:
                f.write(ref_text)
                
            # Convert to MusicXML
            try:
                subprocess.run(["python", "-m", "gelato.data.abc2xml", pred_abc_path, "-o", args.output_dir], check=False, capture_output=True)
                subprocess.run(["python", "-m", "gelato.data.abc2xml", ref_abc_path, "-o", args.output_dir], check=False, capture_output=True)
            except Exception as e:
                print(f"Failed to run abc2xml for {sample_id}: {e}")
                
    # 4. Compute Metrics
    print("Computing metrics...")
    
    # We might need to handle tokenized inputs/outputs differently for metrics if they contain special tokens.
    # The errot_rates expects List[List[int]] for label_ids and preds, which it will parse.
    
    metrics = compute_error_rates(
        tokenizer=tokenizer,
        num_workers=args.num_workers,
        label_ids=references,
        preds=predictions
    )
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Evaluation complete. Metrics saved to {metrics_path}")
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
