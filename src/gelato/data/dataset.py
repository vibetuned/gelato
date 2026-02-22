import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import random
from typing import List, Tuple
from transformers import PreTrainedTokenizer

from .renderer import Renderer
from ..utils.tokenizer import TOK_START, TOK_END, TOK_IMG, TOK_LINE

class GelatoDataset(Dataset):
    def __init__(self, 
                 data_dir: Path, 
                 tokenizer: PreTrainedTokenizer, 
                 max_seq_len: int = 1024,
                 split: str = "train"):
        """
        Expects a directory structure where pre-processed samples might exist, 
        or raw ABC files.
        For simplicity, let's assume `data_dir` contains folders, each with:
         - source.abc
         - source_slice_0.png, source_slice_1.png ... (if pre-processed)
         or just source.abc and we process on the fly? 
         On-the-fly rendering is slow.
         We should assume data is pre-processed into (Image, ABC) pairs.
         
        Let's assume a manifest file or simple folder walk.
        Folder structure:
        data_dir/
          <sample_id>/
             patches/
                0.png
                1.png
                2.png
                3.png
             label.abc (canonicalized)
        
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = self._discover_samples()
        
    def _discover_samples(self):
        # List subdirectories
        samples = []
        if not self.data_dir.exists():
            return []
            
        for d in sorted(self.data_dir.iterdir()):
            if d.is_dir():
                abc_file = d / "label.abc"
                patches_dir = d / "patches"
                if abc_file.exists() and patches_dir.exists():
                    # Check if patches dir is not empty
                    if any(patches_dir.iterdir()):
                         samples.append(d)
        return sorted(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        
        # Load Text
        with open(sample_dir / "label.abc", "r") as f:
            abc_text = f.read().strip()
            
        # Load Images
        patches_dir = sample_dir / "patches"
        patch_files = sorted(list(patches_dir.glob("*.png")), key=lambda p: int(p.stem))[:4] # Force max 4 patches
        
        pixel_values = []
        for p_file in patch_files:
            img = Image.open(p_file).convert("RGB")
            # Assume already resized to 512x512
            pixel_values.append(img)
            
        # Ensure exactly 4 patches by returning them or padding later
        # For now, let's pass them to collator to pad exactly up to 4.
        
        return {
            "abc_text": abc_text,
            "images": pixel_values,
            "id": sample_dir.name
        }

class GelatoCollator:
    def __init__(self, processor, tokenizer, max_length=1024):
        self.processor = processor # SigLIP ImageProcessor
        self.tokenizer = tokenizer # Gemma Tokenizer
        self.max_length = max_length
        
    def __call__(self, batch):
        # batch is list of dicts
        texts = [b["abc_text"] for b in batch]
        images_list = [b["images"] for b in batch] # List of Lists of PIL images
        
        # Process Images
        # SigLIP processor expects [Batch_Size, C, H, W] inputs usually.
        # If we have N patches per sample, we have B*N images.
        
        # Flatten list of lists
        # But first, we need to PAD images if num patches varies.
        
        num_patches_list = [len(l) for l in images_list]
        max_patches = 4 # Enforce exactly 4 patches as per architecture expectations
        
        # Pad images
        padded_images_list = []
        for patches in images_list:
            n = len(patches)
            diff = max_patches - n
            if diff > 0:
                # Create white padding image of same size as first patch
                if n > 0:
                    base_w, base_h = patches[0].size
                else:
                    base_w, base_h = 512, 512 # Fallback
                    
                pad_img = Image.new("RGB", (base_w, base_h), (255, 255, 255))
                patches = patches + [pad_img] * diff
            padded_images_list.append(patches)
            
        # Flatten
        flat_images = [img for sublist in padded_images_list for img in sublist]

        # Use processor
        # process_images returns pixel_values
        image_inputs = self.processor(images=flat_images, return_tensors="pt")
        pixel_values = image_inputs.pixel_values # [B*Max_N, 3, H, W]
        
        # Reshape to [B, Max_N, 3, H, W]
        b = len(batch)
        c, h, w = pixel_values.shape[-3:]
        pixel_values = pixel_values.view(b, max_patches, c, h, w)
        
        # Create image attention mask [B, Max_N]
        # 1 for valid image patch, 0 for padded patch
        image_attention_mask = torch.zeros((b, max_patches), dtype=torch.long)
        for i, patches in enumerate(images_list):
            image_attention_mask[i, :len(patches)] = 1
        
        # Process Text
        # Format: <B> [ABC] <E>
        formatted_texts = [f"{TOK_START} {t} {TOK_END}" for t in texts]
        
        text_inputs = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        )
        
        # Replace padding with -100 in labels so it's ignored in loss
        labels = text_inputs.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "image_attention_mask": image_attention_mask,
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
            "labels": labels
        }
