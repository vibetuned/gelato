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
        num_media_embeds: int = 4, # Patches per image? No, this is for pos emb? 
                                   # Actually Perceiver Resampler takes variable length input. 
        input_dim: int = 1152, # SigLIP 2 Large dimension
        ff_mult: int = 4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, input_dim)) # Position embed for the patches? 
                                                                                       # Usually we add this to input before resampler if we have explicit patches.
                                                                                       # Let's handle positional embeddings outside or simplify.
        
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, Seq_Len, Input_Dim] (e.g. from SigLIP)
        """
        b = x.shape[0]
        
        # Project input to model dim
        x = self.proj_in(x) # [B, S, D]
        
        # Repeat latents for batch
        latents = self.latents.repeat(b, 1, 1) # [B, N, D]
        
        for attn_cross, ff_cross, attn_self, ff_self in self.layers:
            # Cross Attention: Q=Latents, K,V=Input
            latents = attn_cross(latents, x) + latents
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

    def forward(self, x, context=None):
        h = self.heads
        
        q = self.to_q(x)
        context = context if context is not None else x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: t.view(t.shape[0], -1, h, t.shape[-1]//h).transpose(1, 2), (q, k, v))
        
        sim = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = sim.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(out.shape[0], -1, out.shape[-1] * out.shape[-2]) # flatten heads? 
                            # Wait, reshape properly: [B, H, N, D_head] -> [B, N, H*D_head]
        out = out.contiguous().view(out.shape[0], -1, self.heads * (k.shape[-1]))
        
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
