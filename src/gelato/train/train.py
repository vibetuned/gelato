"""Modular Training Script for Gelato VLM model.

Uses Hugging Face Transformers Trainer for robust, standard implementation.
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Add src to sys.path so 'gelato' is importable when run directly
src_dir = str(Path(__file__).resolve().parent.parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
import torch
from transformers import TrainingArguments, Trainer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from gelato.model.models import GelatoModel, GelatoConfig
from gelato.train.utils import get_tokenizer, load_checkpoint
from gelato.train.dataset import GelatoDataset
from gelato.train.utils import GelatoDataCollator

from gelato.model.static import STATICGrammarCompiler, STATICLogitsProcessor
from gelato.train.trainer import GelatoSTATICTrainer # The custom trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--train-config", type=str, default="conf/train.yaml", help="Path to YAML training configuration.")
    conf_args, remaining_argv = conf_parser.parse_known_args()

    yaml_defaults = {}
    if os.path.exists(conf_args.train_config):
        import yaml
        with open(conf_args.train_config, "r") as f:
            yaml_defaults = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser(description="Gelato VLM Trainer", parents=[conf_parser])

    # Data args
    parser.add_argument("--img-dir", type=str, default="data/dataset-small/imgs")
    parser.add_argument("--abc-dir", type=str, default="data/dataset-small/abcs")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=4)

    # Model args
    parser.add_argument("--vision-model-name", type=str, default="convnext_base.dinov3_lvd1689m")
    parser.add_argument("--text-model-name", type=str, default="google/gemma-3-270m")
    parser.add_argument("--text-dim", type=int, default=1024)
    parser.add_argument("--vision-feature-dims", type=int, nargs='+', default=[128, 256, 512])
    parser.add_argument("--engram-warm-start", type=str, default="checkpoints/warm/engram_warm_start.pt", help="Path to warm-start Engram weights.")
    
    # Low memory args
    parser.add_argument("--visual-pool-size", type=int, default=12) # Down from 16
    parser.add_argument("--max-seq-len", type=int, default=768)     # Down from 2048
    parser.add_argument("--grad-accum-steps", type=int, default=8)  # New argument
    parser.add_argument("--mask-ratio", type=float, default=0.5, help="Mask ratio for Engram training")

    # Training args
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--logging-dir", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Auto-detect and resume from the last checkpoint.")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to a specific checkpoint to resume from.")
    parser.add_argument("--phase", type=str, choices=["alignment", "finetune"], default="alignment", 
                        help="Training phase: 'alignment' freezes LLM, 'finetune' trains end-to-end.")
    parser.add_argument("--fine-tune", type=str, default=None, help="Path to checkpoint for fine-tuning phase.")

    parser.set_defaults(**yaml_defaults)
    args, _ = parser.parse_known_args(remaining_argv)
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.resume or args.resume_from_checkpoint:
        existing_runs = sorted(
            [d for d in output_root.iterdir() if d.is_dir() and d.name.startswith("run_")],
            key=lambda d: d.name,
        )
        if existing_runs:
            run_dir = existing_runs[-1]
            logger.info(f"Resuming from existing run: {run_dir}")
        else:
            run_dir = output_root / "run_001"
            run_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"No existing runs found in {output_root}, creating {run_dir}")
    else:
        existing_runs = sorted(
            [d for d in output_root.iterdir() if d.is_dir() and d.name.startswith("run_")],
            key=lambda d: d.name,
        )
        if existing_runs:
            last_num = int(existing_runs[-1].name.split("_")[1])
            next_num = last_num + 1
        else:
            next_num = 1
        run_dir = output_root / f"run_{next_num:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created new run directory: {run_dir}")

    os.environ["TENSORBOARD_LOGGING_DIR"] = args.logging_dir + "/" + run_dir.name

    logger.info("Setting up tokenizer...")
    tokenizer = get_tokenizer(args.text_model_name)

    logger.info("Initializing dataset...")
    train_dataset = GelatoDataset(
        img_dir=args.img_dir,
        abc_dir=args.abc_dir,
        tokenizer=tokenizer,
        input_size=args.input_size,
    )
    
    data_collator = GelatoDataCollator(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len
    )

    logger.info("Initializing model...")
    model_config = GelatoConfig(
        vision_model_name=args.vision_model_name,
        text_model_name=args.text_model_name,
        text_dim=args.text_dim,
        vision_feature_dims=args.vision_feature_dims,
        visual_pool_size=args.visual_pool_size,
        max_seq_len=args.max_seq_len,
        mask_ratio=args.mask_ratio,
    )

    model = GelatoModel(model_config)

    # --- NEW WARM START LOADING LOGIC ---
   
    if args.phase == "finetune":
        logger.info(
            f"Loading weights from {args.fine_tune} for fine-tuning (forcing CPU to save VRAM)..."
        )
        device = torch.device("cpu")
        load_checkpoint(model, args.fine_tune, device=device, eval=False)
        logger.info("🔥 PHASE 2: Unfreezing text model for end-to-end fine-tuning.")
        for param in model.text_model.parameters():
            param.requires_grad = True
    elif args.phase == "alignment":
        logger.info("🥶 PHASE 1: Freezing text model. Training vision projectors only.")
        for param in model.text_model.parameters():
            param.requires_grad = False
            
        # Explicitly ensure the projectors remain unfrozen
        for param in model.mm_projectors.parameters():
            param.requires_grad = True  
            
        # Unfreeze the Engram inserted at layer 1
        for param in model.text_model.model.layers[1].parameters():
            param.requires_grad = True

        warm_start_path = Path(args.engram_warm_start)
        if warm_start_path.exists():
            logger.info(f"Loading Engram warm-start weights from {warm_start_path}...")
            # Load the saved MultiHeadEmbedding state_dict
            state_dict = torch.load(warm_start_path, map_location="cpu", weights_only=True)
            # Inject it into Layer 1's Engram module
            model.text_model.model.layers[1].multi_head_embedding.load_state_dict(state_dict)
            logger.info("✅ Warm-start weights successfully injected!")
        else:
            logger.warning("⚠️ No warm-start weights found. Engram is using random noise!")
    # ------------------------------------

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        run_name=run_dir.name,
        logging_dir=args.logging_dir+"/"+run_dir.name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,  # Important for models expecting multimodal explicit args
        report_to="tensorboard",
        save_total_limit=3,
        logging_first_step=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,  # Usually recommended for modern GPUs like RTX
        dataloader_pin_memory=True,
        # 1. Gradient Accumulation
        # Simulates a batch size of 16 (2 * 8) without eating the RAM of 16 images
        gradient_accumulation_steps=args.grad_accum_steps,
        
        # 2. Gradient Checkpointing
        # Discards intermediate activations during the forward pass and recomputes 
        # them during the backward pass. Saves up to 60% of VRAM!
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        
        # 3. Optimizer Memory
        # Uses 8-bit AdamW to shrink optimizer states from 32-bit to 8-bit
        optim="adamw_8bit",
    )

    logger.info("Compiling STATIC CSR Grammar Tensor...")
    compiler = STATICGrammarCompiler(tokenizer_name=args.text_model_name)
    token_to_state, state_to_allowed = compiler.build_state_tensors(device="cuda")
    static_processor = STATICLogitsProcessor(token_to_state, state_to_allowed)

    trainer = GelatoSTATICTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        static_processor=static_processor,
    )

    checkpoint = None
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint
    elif args.resume:
        last_checkpoint = get_last_checkpoint(str(run_dir))
        if last_checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")
            checkpoint = last_checkpoint
        else:
            logger.warning(f"No checkpoint found in {run_dir}, starting from scratch.")

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    main()
