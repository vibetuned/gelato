## built-in
from typing import List
from dataclasses import dataclass, field
import math
import weakref

## third-party
from sympy import isprime
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex

@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "google/gemma-3-270m" # Updated
    engram_vocab_size: List[int] = field(default_factory=lambda: [20000, 20000])
    #engram_vocab_size: List[int] = field(default_factory=lambda: [256000*5, 256000*5]) # Updated for Gemma
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1]) # Just layer 1 for the 270M model
    pad_id: int = 0 # Gemma's standard pad/eos token
    seed: int = 0
    kernel_size: int = 4
    
@dataclass
class BackBoneConfig:
    hidden_size: int = 640
    vocab_size: int = 256000 # Updated for Gemma
    num_layers: int = 26 # Gemma 3 270M has 26 layers, not 30

engram_cfg = EngramConfig()
backbone_config = BackBoneConfig()

class CompressedTokenizer:
    def __init__(
        self,
        tokenizer_name_or_path,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])
        
        self.lookup_table, self.num_new_token = self._build_lookup_table()
    
    def __len__(self):
        return self.num_new_token
    
    def _build_lookup_table(self):
        old2new = {}
        key2new = {}          
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            
            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
        
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)
    
    def _compress(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        out[pos_mask] = self.lookup_table[valid_ids]
        return out   
    
    def __call__(self, input_ids):
        return self._compress(input_ids)

    def compress_torch(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, 'lookup_tensor'):
            self.lookup_tensor = torch.from_numpy(self.lookup_table).long()
        if self.lookup_tensor.device != input_ids.device:
            self.lookup_tensor = self.lookup_tensor.to(input_ids.device)
            
        pos_mask = input_ids >= 0
        out = input_ids.clone()
        out[pos_mask] = self.lookup_tensor[input_ids[pos_mask]]
        return out

def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1

class NgramHashMapping:
    def __init__(
        self, 
        engram_vocab_size,
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        layer_ids,
        tokenizer_name_or_path,
        pad_id,
        seed,  
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )            
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007
        
        self.layer_multipliers = {}

        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}
        
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1
                
                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start, 
                        seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
            
        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0: return x
            shifted = np.pad(x, ((0, 0), (k, 0)),
                                mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []
        
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            
            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))
        
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers

    def get_ngram_hashes_torch(self, input_ids: torch.Tensor, layer_id: int) -> torch.Tensor:
        B, T = input_ids.shape

        if not hasattr(self, 'layer_multipliers_t'):
            self.layer_multipliers_t = {}
            
        if layer_id not in self.layer_multipliers_t or self.layer_multipliers_t[layer_id].device != input_ids.device:
            self.layer_multipliers_t[layer_id] = torch.tensor(
                self.layer_multipliers[layer_id], dtype=torch.long, device=input_ids.device
            )

        multipliers = self.layer_multipliers_t[layer_id]

        def shift_k(k: int) -> torch.Tensor:
            if k == 0: return input_ids
            shifted = torch.nn.functional.pad(input_ids, (k, 0), value=self.pad_id)[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = torch.bitwise_xor(mix, tokens[k] * multipliers[k])
            
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            
            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash)
        
        return torch.stack(all_hashes, dim=2)

    def hash_torch(self, input_ids: torch.Tensor) -> dict:
        input_ids = self.compressed_tokenizer.compress_torch(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self.get_ngram_hashes_torch(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers

class MultiHeadEmbedding(nn.Module):
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        
        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_input_ids = input_ids + self.offsets
        output = self.embedding(shifted_input_ids)
        
        return output

class ShortConv(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        kernel_size: int = 4, 
        dilation: int = 1, 
        norm_eps: float = 1e-5,
        activation: bool = True,
    ):
        super().__init__()
        self.activation = activation
        
        # Standard Depthwise 1D Convolution
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        # Single normalization layer for the dense hidden state
        self.norm = nn.RMSNorm(hidden_size, eps=norm_eps) 
        
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, L, D)
        Output: (B, L, D)
        """
        B, L, D = x.shape
        
        x_norm = self.norm(x)
        # Conv1d expects (Batch, Channels, Length)
        x_bct = x_norm.transpose(1, 2)
        
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :L] # Crop causal padding

        if self.activation:
            y_bct = self.act_fn(y_bct)
            
        # Transpose back to (Batch, Length, Dimension)
        y = y_bct.transpose(1, 2).contiguous()
        
        return y

class Engram(nn.Module):
    def __init__(self, layer_id, original_layer, hidden_size=640, mask_ratio=0.5):
        super().__init__()
        self.layer_id = layer_id
        self.original_layer = original_layer
        self.hidden_size = hidden_size
        self.mask_ratio = mask_ratio
        
        # Keep hash mapping and embedding standard
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size = engram_cfg.max_ngram_size,
            n_embed_per_ngram = engram_cfg.n_embed_per_ngram,
            n_head_per_ngram = engram_cfg.n_head_per_ngram,
            layer_ids = engram_cfg.layer_ids,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            pad_id = engram_cfg.pad_id,
            seed = engram_cfg.seed,
        )
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y],
            D = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
        )
        
        self.short_conv = ShortConv(
            hidden_size = self.hidden_size,
            kernel_size = engram_cfg.kernel_size,
            dilation    = engram_cfg.max_ngram_size,
        )
        
        engram_hidden_size = (engram_cfg.max_ngram_size-1) * engram_cfg.n_embed_per_ngram
        
        # Single projections instead of ModuleLists
        self.value_proj = nn.Linear(engram_hidden_size, self.hidden_size)
        self.key_proj = nn.Linear(engram_hidden_size, self.hidden_size)
        self.norm1 = nn.RMSNorm(self.hidden_size)
        self.norm2 = nn.RMSNorm(self.hidden_size)

        self.clear_cache()
    
    def clear_cache(self):
        """Resets the state for a new generation or training step."""
        self._conv_cache = None
        self._token_cache = None
        self._vis_seq_len = 0
        
    def set_prefill_state(self, input_ids, vis_seq_len):
        """Called by the model during the first forward pass."""
        self._vis_seq_len = vis_seq_len
        self._conv_cache = None

        if self.training and self.mask_ratio > 0:
            # Randomly replace tokens with pad so n-gram hashes become incomplete
            mask = torch.rand(input_ids.shape, device=input_ids.device) < self.mask_ratio
            # Never mask the first token (BOS) — keep at least some anchor
            mask[:, 0] = False
            masked_ids = input_ids.clone()
            masked_ids[mask] = self.hash_mapping.pad_id
            self._token_cache = masked_ids
        else:
            self._token_cache = input_ids
        
    def append_decode_token(self, next_token_id):
        """Called by the model during autoregressive generation."""
        if self._token_cache is not None:
            self._token_cache = torch.cat([self._token_cache, next_token_id], dim=1)

    def forward(self, hidden_states, **kwargs):
        # Fallback safeguard: if state isn't set, just run the normal layer
        if self._token_cache is None:
            return self.original_layer(hidden_states, **kwargs) if self.original_layer else (hidden_states,)

        L_vis = self._vis_seq_len
        L_total = hidden_states.shape[1]
        
        # Phase Detection
        cache_len = 0
        if hasattr(kwargs.get('past_key_values', None), 'get_seq_length'):
            cache_len = kwargs['past_key_values'].get_seq_length()

        is_prefill = (L_total > 1) or (cache_len == 0)
        is_decode = (L_total == 1) and (cache_len > 0)

        if not (is_prefill or is_decode):
            return self.original_layer(hidden_states, **kwargs) if self.original_layer else (hidden_states,)

        if is_prefill:
            vis_states  = hidden_states[:, :L_vis]
            text_states = hidden_states[:, L_vis:]
            target_ids = self._token_cache 
        else:
            vis_states = None
            text_states = hidden_states
            target_ids = self._token_cache # Full rolling history is kept in the cache

        # 1. Retrieve N-gram embeddings using the clean target_ids
        hash_input_ids = self.hash_mapping.hash_torch(target_ids)[self.layer_id]
        
        if is_decode:
            hash_input_ids = hash_input_ids[:, -1:]

        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)

        # 2. Compute keys and queries
        key        = self.key_proj(embeddings)
        normed_key = self.norm1(key)
        query      = text_states
        normed_query = self.norm2(query)

        # 3. Context-Aware Gate  (paper: α_t = σ(RMSNorm(h)ᵀ RMSNorm(k) / √d))
        gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.hidden_size)
        gate = gate.sigmoid().unsqueeze(-1)

        # 4. Compute Value
        value = gate * self.value_proj(embeddings)

        # 5. THE FIX: Rolling Convolution Cache
        if is_prefill:
            text_output = value + self.short_conv(value)
            # Initialize the state cache
            self._conv_cache = value 
        else:
            if hasattr(self, '_conv_cache') and self._conv_cache is not None:
                # Append the newest token to the history
                self._conv_cache = torch.cat([self._conv_cache, value], dim=1)
                
                # Calculate the exact receptive field needed
                kernel = self.short_conv.conv.kernel_size[0]
                dilation = self.short_conv.conv.dilation[0]
                receptive_field = (kernel - 1) * dilation + 1
                
                # Slice the rolling window and trim the cache to prevent memory leak
                cache_window = self._conv_cache[:, -receptive_field:]
                self._conv_cache = cache_window
                
                # Run the conv and extract only the newest token's output
                window_out = self.short_conv(cache_window)
                text_output = value + window_out[:, -1:]
            else:
                # Fallback safeguard
                text_output = value + self.short_conv(value)

        # 6. Residual connection
        augmented_text = text_states + text_output

        # Recombine based on phase
        if is_prefill:
            augmented_hidden = torch.cat([vis_states, augmented_text], dim=1)
        else:
            augmented_hidden = augmented_text

        if self.original_layer is not None:
            return self.original_layer(augmented_hidden, **kwargs)
            
        return (augmented_hidden,)

