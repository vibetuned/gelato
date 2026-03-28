from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import PreTrainedTokenizer

class LetterboxResize:
    """Resize maintaining aspect ratio, pad the short side to target size."""
    def __init__(self, target_size=640, fill=(255, 255, 255)):
        self.target_size = target_size
        self.fill = fill  # White background matches score paper

    def __call__(self, img):
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Create white canvas and paste the resized image centered
        canvas = Image.new("RGB", (self.target_size, self.target_size), self.fill)
        offset_x = (self.target_size - new_w) // 2
        offset_y = (self.target_size - new_h) // 2
        canvas.paste(img, (offset_x, offset_y))
        
        return canvas

class GelatoDataset(Dataset):
    """
    Dataset for VLM training. Pairs an image with its corresponding ABC notation.
    Text is wrapped with the tokenizer's native BOS/EOS tokens.
    """
    def __init__(self, img_dir, abc_dir, tokenizer: PreTrainedTokenizer, input_size=640):
        self.img_dir = Path(img_dir)
        self.abc_dir = Path(abc_dir)
        self.bos = tokenizer.bos_token
        self.eos = tokenizer.eos_token

        self.img_files = sorted([
            f for f in self.img_dir.iterdir()
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
        ])

        # Standard normalization for TIMM vision encoders
        self.transform = transforms.Compose([
            LetterboxResize(target_size=input_size, fill=(255, 255, 255)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        abc_path = self.abc_dir / f"{img_path.stem}.abc"

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image) # TIMM standard normalization

        with open(abc_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Just return the pure text. We will handle tokens in the collator.
        return {
            "pixel_values": pixel_values,
            "text": text
        }
