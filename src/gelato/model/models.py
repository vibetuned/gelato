
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, AutoModelForCausalLM, GenerationMixin
from .layers import Engram
import timm

class GelatoConfig(PretrainedConfig):
    model_type = "gelato"
    def __init__(self,
                 vision_model_name="convnext_base.dinov3_lvd1689m",
                 text_model_name="google/gemma-3-270m",
                 text_dim=1024,
                 vision_feature_dims=[128, 256, 512],
                 visual_pool_size=16,   # each feature map is pooled to (P x P) tokens
                 max_seq_len=2048,
                 mask_ratio=0.5,
                 num_hidden_layers=26,  # <-- Add this to satisfy HF generation
                 vocab_size=191,     # <-- Good practice to add this too for Gemma 3
                 **kwargs):
        super().__init__(**kwargs)
        self.vision_model_name = vision_model_name
        self.text_model_name = text_model_name
        self.text_dim = text_dim
        self.vision_feature_dims = vision_feature_dims
        self.visual_pool_size = visual_pool_size
        self.max_seq_len = max_seq_len
        self.mask_ratio = mask_ratio
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.engram = True


class GelatoModel(PreTrainedModel, GenerationMixin):
    config_class = GelatoConfig
    main_input_name = "input_ids"

    _keys_to_ignore_on_load_missing = ["text_model.lm_head.weight"]

    def __init__(self, config: GelatoConfig):
        super().__init__(config)

        # Visual token budget: P*P per scale × num_scales
        # e.g. 16*16 * 3 = 768 tokens
        self._num_visual_tokens = config.visual_pool_size ** 2 * len(config.vision_feature_dims)

        # 1. Vision Encoder (timm ConvNeXt DINOv3)
        self.vision_model = timm.create_model(
            config.vision_model_name,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2)
        )
        for param in self.vision_model.parameters():
            param.requires_grad = False
        self.vision_model.eval()

        # Adaptive pools: reduce each feature map to (visual_pool_size × visual_pool_size)
        # before flattening, so the visual token count is independent of input resolution (fix #6)
        self.visual_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(config.visual_pool_size)
            for _ in config.vision_feature_dims
        ])

        # 2. Text Decoder (Gemma 3) — position budget must cover visual + text tokens (fix #6)
        max_pos = self._num_visual_tokens + config.max_seq_len
        print(f"Loading Text Decoder: {config.text_model_name}")
        self.text_model = AutoModelForCausalLM.from_pretrained(
            config.text_model_name,
            max_position_embeddings=max_pos,
            ignore_mismatched_sizes=True
        )

        real_text_dim = self.text_model.config.hidden_size
        self.config.text_dim = real_text_dim # Overwrite the config for consistency
        self.config.vocab_size = self.text_model.config.vocab_size 
        model_dtype = self.text_model.dtype

        # 3. Multi-Scale Projectors (one per feature map)
        self.mm_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, real_text_dim),
                nn.GELU(),
                nn.Linear(real_text_dim, real_text_dim)
            ) for dim in config.vision_feature_dims
        ]).to(model_dtype)

        # 4. Engram Surgery — parent_model reference lets Engram read cached input_ids (fix #1)
        if config.engram:
            original_layer = self.text_model.model.layers[1]
            engram_layer = Engram(layer_id=1, original_layer=original_layer, hidden_size=real_text_dim, mask_ratio=config.mask_ratio).to(model_dtype)
            
            # Duck-type the Engram module to keep Hugging Face's forward loop happy
            if hasattr(original_layer, "attention_type"):
                engram_layer.attention_type = original_layer.attention_type
            
            self.text_model.model.layers[1] = engram_layer

    def train(self, mode=True):
        super().train(mode)
        # Keep vision encoder firmly in eval mode regardless of training state
        self.vision_model.eval()
        return self

    def _extract_visual_features(self, pixel_values):
        """Shared helper: vision encode → pool → project → concatenate."""
        with torch.no_grad():
            feature_maps = self.vision_model(pixel_values)

        tokens = []
        for i, feat in enumerate(feature_maps):
            pooled    = self.visual_pools[i](feat)      # [B, C, P, P]
            flattened = pooled.flatten(2)               # [B, C, P*P]
            sequence  = flattened.transpose(1, 2)       # [B, P*P, C]
            projected = self.mm_projectors[i](sequence) # [B, P*P, D]
            tokens.append(projected)

        return torch.cat(tokens, dim=1)  # [B, num_visual_tokens, D]

    def forward(self, pixel_values, input_ids, labels=None, attention_mask=None, past_key_values=None, use_cache=None, **kwargs):
        b = input_ids.shape[0]

        is_prefill = (
            past_key_values is None 
            or (hasattr(past_key_values, 'get_seq_length') 
                and past_key_values.get_seq_length() == 0)
        )

        # PHASE 1: PREFILL (past_key_values is None)
        if is_prefill:
            visual_features = self._extract_visual_features(pixel_values)
            vis_seq_len = visual_features.shape[1]

            inputs_embeds = self.text_model.get_input_embeddings()(input_ids)
            combined_embeds = torch.cat([visual_features, inputs_embeds], dim=1)

            if attention_mask is not None:
                vis_mask = torch.ones((b, vis_seq_len), device=combined_embeds.device)
                attention_mask = torch.cat([vis_mask, attention_mask], dim=1)

            if labels is not None:
                vis_labels = torch.full((b, vis_seq_len), -100, dtype=labels.dtype, device=labels.device)
                labels = torch.cat([vis_labels, labels], dim=1)

            # Explicitly pass the clean prefill state to the Engram layer
            if self.config.engram:
                self.text_model.model.layers[1].set_prefill_state(input_ids=input_ids, vis_seq_len=vis_seq_len)

            return self.text_model(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                labels=labels,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs
            )
            
        # PHASE 2: DECODE (past_key_values exists)
        else:
            # Tell the Engram layer to append the new token
            if self.config.engram:
                self.text_model.model.layers[1].append_decode_token(input_ids)

            # --- YOUR FIX: Aligning the Attention Mask ---
            if attention_mask is not None:
                # Use self._num_visual_tokens to match the static cache size
                vis_mask = torch.ones((b, self._num_visual_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([vis_mask, attention_mask], dim=1) 
            # ---------------------------------------------

            return self.text_model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs
            )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        """This is called natively by Hugging Face's generate loop at every step."""
        pv = kwargs.get("pixel_values")
        if pv is None:
            print("WARNING: pixel_values is None in prepare_inputs_for_generation")
        elif past_key_values is None:
            print(f"PREFILL: pixel_values shape = {pv.shape}")
        
        if past_key_values is not None:
            # In the decode phase, only pass the single newest token to the forward pass
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "pixel_values": kwargs.get("pixel_values"), # Crucial: keep passing the image through
            "use_cache": kwargs.get("use_cache", True),
        }

    def generate(self, pixel_values, **kwargs):
        """Clean, native generation wrapper."""
        # 1. Reset the Engram state before starting a new generation
        if self.config.engram:
            self.text_model.model.layers[1].clear_cache()
        
        # 2. Build the initial prompt (just the BOS token)
        b = pixel_values.shape[0]
        bos_token_id = self.text_model.config.bos_token_id
        input_ids = torch.full((b, 1), bos_token_id, dtype=torch.long, device=pixel_values.device)

        # 3. --- THE FIX: Explicitly pass Gemma's stopping criteria ---
        if "eos_token_id" not in kwargs:
            kwargs["eos_token_id"] = self.text_model.config.eos_token_id
        if "pad_token_id" not in kwargs:
            kwargs["pad_token_id"] = self.text_model.config.pad_token_id

        # 4. Delegate directly to Hugging Face's highly optimized generate loop
        return super().generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            **kwargs
        )
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Routes the Trainer's gradient checkpointing activation 
        down to the underlying Gemma 3 text model.
        """
        self.text_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        """
        Routes the Trainer's gradient checkpointing deactivation.
        """
        self.text_model.gradient_checkpointing_disable()