import torch
import torch.nn as nn
import math

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        depth: int = 3,  # Standard is often 6, but plan implies simple flamenco style? Plan doesn't specify depth.
                        # Flamingo used 6. We'll stick to a reasonable default.
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 256,
        num_media_embeds: int = 4, # Max patches per image
        input_dim: int = 1152, # SigLIP 2 Large dimension
        ff_mult: int = 4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, input_dim)) # Position embed for the patches
        
        self.proj_in = nn.Linear(input_dim, dim) if input_dim != dim else nn.Identity()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Perceiver Attention: Latents attend to Media
                Attention(dim, dim_head=dim_head, heads=heads), 
                FeedForward(dim, mult=ff_mult),
                # Self Attention: Latents attend to Latents
                Attention(dim, dim_head=dim_head, heads=heads),
                FeedForward(dim, mult=ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, num_media: int = None, image_attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [Batch, Seq_Len, Input_Dim] (e.g. from SigLIP)
        num_media: The actual number of image patches mapped to this seq len.
        """
        b = x.shape[0]
        
        if num_media is not None:
            # x is [B, Num_Patches * Patch_Seq_Len, Input_Dim]
            patch_seq_len = x.shape[1] // num_media
            # Add positional embeddings for the media
            # media_pos_emb is [Num_Media, 1, Input_Dim]
            pos_emb = self.media_pos_emb[:num_media].repeat(1, patch_seq_len, 1) # [Num_Media, Patch_Seq_Len, Input_Dim]
            pos_emb = pos_emb.view(-1, x.shape[-1]) # [Num_Media * Patch_Seq_Len, Input_Dim]
            x = x + pos_emb.unsqueeze(0) # [B, Seq_Len, Input_Dim]
        
        # Project input to model dim
        x = self.proj_in(x) # [B, S, D]
        
        # Build attention mask for cross-attention
        cross_attn_mask = None
        if image_attention_mask is not None and num_media is not None:
            # image_attention_mask: [B, Num_Media] (1 for valid, 0 for padding)
            # Expand to [B, Num_Media * Patch_Seq_Len]
            expanded_mask = image_attention_mask.unsqueeze(-1).repeat(1, 1, patch_seq_len).view(b, -1)
            # Boolean mask for attention computation (True where padding)
            cross_attn_mask = (expanded_mask == 0) # [B, Seq_Len]
        
        # Repeat latents for batch
        latents = self.latents.repeat(b, 1, 1) # [B, N, D]
        
        for attn_cross, ff_cross, attn_self, ff_self in self.layers:
            # Cross Attention: Q=Latents, K,V=Input
            latents = attn_cross(latents, context=x, mask=cross_attn_mask) + latents
            latents = ff_cross(latents) + latents
            
            # Self Attention: Q,K,V=Latents
            latents = attn_self(latents) + latents
            latents = ff_self(latents) + latents
            
        return self.norm(latents)

class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads
        
        q = self.to_q(x)
        context = context if context is not None else x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: t.view(t.shape[0], -1, h, t.shape[-1]//h).transpose(1, 2), (q, k, v))
        
        sim = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if mask is not None:
            # mask is [B, Seq_Len] -> True where padding
            # Expand to [B, 1, 1, Seq_Len]
            mask = mask.unsqueeze(1).unsqueeze(1)
            sim = sim.masked_fill(mask, -torch.finfo(sim.dtype).max)
            
        attn = sim.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        # [B, H, N, D_head] -> [B, N, H, D_head] -> [B, N, H*D_head]
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.shape[0], out.shape[1], -1)
        
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)
