"""Evaluation script for Gelato VLM model.

Loads a checkpoint, runs generation on a dataset, and reports SER/CER/LER.
"""

import sys
import os
import logging
import argparse
from pathlib import Path
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

src_dir = str(Path(__file__).resolve().parent.parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import LogitsProcessorList, set_seed
from tqdm import tqdm

from gelato.model.models import GelatoModel, GelatoConfig
from gelato.train.utils import get_tokenizer, load_checkpoint, GelatoDataCollator
from gelato.train.dataset import GelatoDataset
from gelato.model.static import ABCGrammarCompiler, ABCLogitsProcessor
from gelato.metrics.error_rates import compute_error_rates

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--eval-config", type=str, default="conf/eval.yaml")
    conf_args, remaining_argv = conf_parser.parse_known_args()

    yaml_defaults = {}
    if os.path.exists(conf_args.eval_config):
        import yaml
        with open(conf_args.eval_config, "r") as f:
            yaml_defaults = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser(description="Gelato VLM Evaluator", parents=[conf_parser])

    # Data args
    parser.add_argument("--img-dir", type=str, default="data/dataset-small/imgs")
    parser.add_argument("--abc-dir", type=str, default="data/dataset-small/abcs")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)

    # Model args
    parser.add_argument("--vision-model-name", type=str, default="convnext_base.dinov3_lvd1689m")
    parser.add_argument("--text-model-name", type=str, default="google/gemma-3-270m")
    parser.add_argument("--text-dim", type=int, default=1024)
    parser.add_argument("--vision-feature-dims", type=int, nargs='+', default=[128, 256, 512])
    parser.add_argument("--visual-pool-size", type=int, default=12)
    parser.add_argument("--max-seq-len", type=int, default=768)

    # Eval args
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to evaluate.")
    parser.add_argument("--tb-dir", type=str, default="runs", help="Root TensorBoard log directory.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-static", action="store_true", help="Disable STATIC grammar constraint during generation.")
    parser.add_argument("--num-workers-metric", type=int, default=8, help="Workers for Levenshtein computation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.set_defaults(**yaml_defaults)
    args, _ = parser.parse_known_args(remaining_argv)
    return args


@torch.inference_mode()
def run_eval(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    logger.info("Setting up tokenizer...")
    tokenizer = get_tokenizer(args.text_model_name)

    logger.info("Initializing dataset...")
    dataset = GelatoDataset(
        img_dir=args.img_dir,
        abc_dir=args.abc_dir,
        tokenizer=tokenizer,
        input_size=args.input_size,
    )

    collator = GelatoDataCollator(tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(args.device == "cuda"),
    )

    logger.info("Initializing model...")
    model_config = GelatoConfig(
        vision_model_name=args.vision_model_name,
        text_model_name=args.text_model_name,
        text_dim=args.text_dim,
        vision_feature_dims=args.vision_feature_dims,
        visual_pool_size=args.visual_pool_size,
        max_seq_len=args.max_seq_len,
    )
    model = GelatoModel(model_config)
    load_checkpoint(model, args.checkpoint, device=device, eval=True)
    model.to(device)

    logits_processor = None
    if not args.no_static:
        logger.info("Compiling ABC CSR Grammar Tensor...")
        compiler = ABCGrammarCompiler(tokenizer=tokenizer)
        t2s, s2a, _ = compiler.build_state_tensors(device="cuda")
        static_processor = ABCLogitsProcessor(t2s, s2a, tokenizer)
        logits_processor = LogitsProcessorList([static_processor])

    all_preds = []   # list of token-id lists
    all_labels = []  # list of token-id lists

    logger.info(f"Running evaluation on {len(dataset)} samples...")
    #model.config.engram = False
    pbar = tqdm(dataloader, desc="Evaluating", unit="batch")
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]  # (B, seq_len), padding = -100

        #out = model(
        #    pixel_values=pixel_values,
        #    input_ids=batch["input_ids"].to(device),
        #    labels=labels.to(device),
        #)
        #print(f"Loss WITHOUT Engram: {out.loss.item()}")

        generated = model.generate(
            pixel_values=pixel_values,
            logits_processor=logits_processor,
            max_new_tokens=args.max_new_tokens,
        )  # (B, gen_len), on device

        generated = generated.cpu()

        # Strip padding (-100) from labels to get clean reference token ids
        for i in range(labels.shape[0]):
            ref = labels[i][labels[i] != -100].tolist()
            pred = generated[i].tolist()
            
            # --- THE ALIGNMENT FIX ---
            bos_id = tokenizer.bos_token_id
            eos_id = tokenizer.eos_token_id
            pad_id = tokenizer.pad_token_id
            
            # Remove BOS, EOS, and PAD from reference so it's pure text
            ref = [t for t in ref if t not in (bos_id, eos_id, pad_id)]
            
            # Remove EOS and PAD from prediction (pred doesn't have BOS because of inputs_embeds)
            pred = [t for t in pred if t not in (bos_id, eos_id, pad_id)]
            
            # -------------------------
            logger.info(f"Ref: {tokenizer.decode(ref)}")
            logger.info(f"Pred: {tokenizer.decode(pred)}")
            
            all_labels.append(ref)
            all_preds.append(pred)

        pbar.set_postfix(samples=len(all_preds))

    logger.info("Computing error rates...")
    metrics = compute_error_rates(
        tokenizer=tokenizer,
        num_workers=args.num_workers_metric,
        label_ids=all_labels,
        preds=all_preds,
    )

    logger.info("=" * 40)
    logger.info("Evaluation Results")
    logger.info("=" * 40)
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}%")
    logger.info("=" * 40)

    checkpoint_name = Path(args.checkpoint).name
    tb_log_dir = os.path.join(args.tb_dir, checkpoint_name)
    logger.info(f"Writing metrics to TensorBoard at {tb_log_dir}")
    writer = SummaryWriter(log_dir=tb_log_dir)
    for name, value in metrics.items():
        writer.add_scalar(f"eval/{name}", value)
    writer.close()

    return metrics


def main():
    args = parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
