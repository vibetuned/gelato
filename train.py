import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
import transformers
from transformers import Trainer, TrainingArguments, SiglipImageProcessor, AutoTokenizer

from gelato.model.modeling import GelatoModel, GelatoConfig
from gelato.data.dataset import GelatoDataset, GelatoCollator
from gelato.utils.tokenizer import get_tokenizer, resize_model_embeddings, TOK_START, TOK_END, TOK_IMG, TOK_LINE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    vision_model_name: str = field(default="google/siglip2-large-patch16-512")
    text_model_name: str = field(default="google/gemma-3-270m")
    resampler_depth: int = field(default=3)
    max_seq_len: int = field(default=2048)

@dataclass
class DataArguments:
    data_dir: str = field(default="./data/processed")

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 1. Initialize Tokenizer & Processor
    tokenizer = get_tokenizer(model_args.text_model_name)
    processor = SiglipImageProcessor.from_pretrained(model_args.vision_model_name)
    
    # 2. Initialize Model
    config = GelatoConfig(
        vision_model_name=model_args.vision_model_name,
        text_model_name=model_args.text_model_name,
        resampler_depth=model_args.resampler_depth,
        max_seq_len=model_args.max_seq_len
    )
    model = GelatoModel(config)
    
    # Resize embeddings for special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # 3. Load Dataset
    dataset = GelatoDataset(
        data_dir=Path(data_args.data_dir), 
        tokenizer=tokenizer,
        max_seq_len=model_args.max_seq_len,
        split="train"
    )
    
    if len(dataset) == 0:
        logger.warning(f"No samples found in {data_args.data_dir}. Ensure you have generated data.")
    
    collator = GelatoCollator(processor, tokenizer, max_length=model_args.max_seq_len)
    
    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    
    # If no action is specified, default to training
    if not (training_args.do_train or training_args.do_eval or training_args.do_predict):
        logger.info("No action flag provided. Defaulting to --do_train.")
        training_args.do_train = True

    # Important: prevent Trainer from filtering out 'abc_text' which is needed by collator
    training_args.remove_unused_columns = False
    
    # Disable safetensors saving to avoid shared weight issues (Gemma ties weights)
    training_args.save_safetensors = False

    # Auto-resume logic
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
            from transformers.trainer_utils import get_last_checkpoint
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is not None:
                logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")
                checkpoint = last_checkpoint
        
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

if __name__ == "__main__":
    train()
