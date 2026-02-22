import torch
import torch.nn as nn
from transformers import PreTrainedModel, SiglipVisionModel, Gemma2ForCausalLM, Gemma2Config, SiglipVisionConfig
# Note: Gemma 3 is released as Gemma2ForCausalLM architecture usually, or we use AutoModel.
# The user specified "Gemma 3 270M".
# If it's not available in standard transformers yet, we might fallback to Gemma 2 logic.
# But for now let's assume standard causal LM interface.

from .resampler import PerceiverResampler
# from ..utils.tokenizer import IMG_TOKEN_ID # Removed dependency on constant ID


from transformers import PretrainedConfig

class GelatoConfig(PretrainedConfig):
    # Minimal config wrapper
    model_type = "gelato"
    def __init__(self, 
                 vision_model_name="google/siglip2-large-patch16-512",
                 text_model_name="google/gemma-3-270m",
                 resampler_depth=3,
                 resampler_latents=256,
                 resampler_dim=1024, # Match gemma hidden size
                 vision_dim=1152,
                 max_seq_len=2048,
                 **kwargs):   # Match siglip hidden size
        super().__init__(**kwargs)
        self.vision_model_name = vision_model_name
        self.text_model_name = text_model_name
        self.resampler_depth = resampler_depth
        self.resampler_latents = resampler_latents
        self.resampler_dim = resampler_dim
        self.vision_dim = vision_dim
        self.max_seq_len = max_seq_len

    def to_dict(self):
        d = super().to_dict()
        # Ensure our custom fields are included nicely, PretrainedConfig automatically copies self.attr 
        # but just in case:
        d.update({
            "vision_model_name": self.vision_model_name,
            "text_model_name": self.text_model_name,
            "resampler_depth": self.resampler_depth,
            "resampler_latents": self.resampler_latents,
            "resampler_dim": self.resampler_dim,
            "vision_dim": self.vision_dim,
            "max_seq_len": self.max_seq_len,
        })
        return d

class GelatoModel(nn.Module):
    def __init__(self, config: GelatoConfig):
        super().__init__()
        self.config = config
        
        # Vision Encoder (Frozen)
        print(f"Loading Vision Encoder: {config.vision_model_name}")
        self.vision_model = SiglipVisionModel.from_pretrained(config.vision_model_name)
        for param in self.vision_model.parameters():
            param.requires_grad = False
        self.vision_model.eval() # Ensure eval mode initially
            
        # Text Decoder (Trainable)
        print(f"Loading Text Decoder: {config.text_model_name}")
        # Using AutoModelForCausalLM is safer
        from transformers import AutoModelForCausalLM
        
        # Pass max_position_embeddings to config override memory usage
        self.text_model = AutoModelForCausalLM.from_pretrained(
            config.text_model_name,
            max_position_embeddings=config.max_seq_len,
            ignore_mismatched_sizes=True # In case position embeddings are resized
        )
        
        # Determine dimensions dynamically
        vision_dim = getattr(self.vision_model.config, "hidden_size", config.vision_dim)
        text_dim = getattr(self.text_model.config, "hidden_size", config.resampler_dim)
        
        # Connector (Trainable)
        self.resampler = PerceiverResampler(
            dim=text_dim,
            depth=config.resampler_depth,
            num_latents=config.resampler_latents,
            input_dim=vision_dim,
            num_media_embeds=64  # Safely handle up to 64 image patches instead of 4
        )
        
    def train(self, mode=True):
        super().train(mode)
        # Keep vision encoder in eval mode (frozen)
        self.vision_model.eval()
        return self
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.text_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        
    def gradient_checkpointing_disable(self):
        self.text_model.gradient_checkpointing_disable()
        
    def resize_token_embeddings(self, new_num_tokens):
        self.text_model.resize_token_embeddings(new_num_tokens)

    def forward(self, pixel_values, input_ids, image_attention_mask=None, labels=None, attention_mask=None):
        """
        pixel_values: [B, Num_Patches, C, H, W] or [B, C, H, W] if single patch?
                      Plan says [Batch, 4, 3, 512, 512].
        input_ids: [B, Seq_Len]
                    Contains <I> tokens where visual embeddings should be injected?
                    Or we prefix visual embeddings.
                    Plan says: <I>...<I> <B> [ABC] <E>
                    Ideally we have N <I> tokens corresponding to N latents?
                    Or just inject embeddings at the start.
        """
        b, num_patches, c, h, w = pixel_values.shape
        # Flatten patches for encoder: [B*N, C, H, W]
        # pixel_values = pixel_values.view(-1, c, h, w)
        
        # Wait, Siglip takes [B, C, H, W].
        # We process each patch or the whole image?
        # Plan: "Splits each segment into N patches".
        # So effective batch size for vision is B * N.
        
        with torch.no_grad():
            vision_outputs = self.vision_model(pixel_values.view(b * num_patches, c, h, w))
            image_embeds = vision_outputs.last_hidden_state # [B*N, H*W, D_vis]
        
        # Reshape back to [B, N, Seq, D_vis] -> [B, N*Seq, D_vis]
        # Actually Perceiver Resampler might want to look at all patches at once.
        # [B, N * (H*W), D_vis]
        image_embeds = image_embeds.view(b, num_patches * image_embeds.shape[1], -1)
        
        # Pass through Resampler
        # Output: [B, Num_Latents, D_txt]
        visual_features = self.resampler(
            image_embeds, 
            num_media=num_patches,
            image_attention_mask=image_attention_mask
        )
        
        # Combine with Text Embeddings
        # We need to replace special tokens <I> with these features or concat?
        # Simpler approach: Concat [Visual_Features, Text_Embeddings]
        # But we need input_ids to NOT contain the visual part then?
        # Or we use inputs_embeds argument of Gemma.
        
        # Get text embeddings
        inputs_embeds = self.text_model.get_input_embeddings()(input_ids) # [B, S, D_txt]
        
        # We assume input_ids starts with placeholders or we just prepend.
        # Plan: "<I>...<I> <B> [ABC] <E>"
        # If we have fixed number of latents (256), we should have 256 <I> tokens?
        # Or we just prepend visual features and ignore the <I> tokens in input_ids (remove them).
        
        # Let's say we prepend.
        # inputs_embeds = torch.cat([visual_features, inputs_embeds], dim=1)
        # attention_mask = ... adjust mask ...
        
        # But we need labels aligned.
        # If we prepend, we need to shift labels.
        
        # Alternative: The dataset provides input_ids with PLACEHOLDERS for the image latents.
        # We replace the embeddings at those positions.
        # This is cleaner if we want flexible interleaving (though here it's just prefix).
        
        # For now, let's implement the "Prepend" strategy as it's standard for captioning.
        # The dataset should yield input_ids for the text ONLY (starting with <B>).
        # We concat visual latents at the front.
        
        # Construct final embeddings
        combined_embeds = torch.cat([visual_features, inputs_embeds], dim=1)
        
        # Construct attention mask
        if attention_mask is None:
            attention_mask = torch.ones(combined_embeds.shape[:2], device=combined_embeds.device)
        else:
            # Add ones for visual part
            vis_mask = torch.ones((b, visual_features.shape[1]), device=combined_embeds.device)
            attention_mask = torch.cat([vis_mask, attention_mask], dim=1)
            
        # Construct labels
        if labels is not None:
            # Pad labels with -100 for visual part
            vis_labels = torch.full((b, visual_features.shape[1]), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([vis_labels, labels], dim=1)
            
        outputs = self.text_model(inputs_embeds=combined_embeds, attention_mask=attention_mask, labels=labels)
        return outputs

    def generate(self, pixel_values, image_attention_mask=None, prompt_ids=None, **kwargs):
        # Inference logic
        b, num_patches, c, h, w = pixel_values.shape
        with torch.no_grad():
            vision_outputs = self.vision_model(pixel_values.view(b * num_patches, c, h, w))
            image_embeds = vision_outputs.last_hidden_state
        
        image_embeds = image_embeds.view(b, num_patches * image_embeds.shape[1], -1)
        visual_features = self.resampler(
            image_embeds,
            num_media=num_patches,
            image_attention_mask=image_attention_mask
        )
        
        # We can't easily use .generate directly if we need to modify inputs_embeds unless we assume inputs_embeds is supported?
        # Gemma .generate supports inputs_embeds? Usually yes via underlying model.
        # But for the first step only.
        # We need to pass the visual features as a "prompt".
        
        # Actually easier: Create a wrapper forward for generate?
        # Or just use the model as a causal LM where the first N tokens are forced.
        
        # Transformer's `generate` doesn't support passing pure embeddings easily as "past key values" without specific handling.
        # But we can pass `inputs_embeds`.
        
        # For the first step we pass `inputs_embeds = visual_features + prompt_embeddings`.
        # `generate` handles the loop.
        
        # Helper to get embeddings for prompt
        if prompt_ids is None:
             # Default prompt: <B>
             # We rely on tokenizer from outside.
             raise ValueError("prompt_ids must be provided")

        prompt_embeds = self.text_model.get_input_embeddings()(prompt_ids)
        combined_embeds = torch.cat([visual_features, prompt_embeds], dim=1)
        
        return self.text_model.generate(inputs_embeds=combined_embeds, **kwargs)
