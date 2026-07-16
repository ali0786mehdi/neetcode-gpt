import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # First call: initialize cache
        if self.cache_k is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            # Append along sequence dimension
            self.cache_k = torch.cat([self.cache_k, new_k], dim=1)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=1)

        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None


class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None
    ) -> Tuple[torch.Tensor, KVCache]:

        # 1. Project input to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Create cache if needed
        if kv_cache is None:
            kv_cache = KVCache()

        # 3. Update cache
        cached_k, cached_v = kv_cache.update(k, v)

        # 4. Scaled Dot-Product Attention
        d_k = q.size(-1)

        scores = torch.matmul(q, cached_k.transpose(-2, -1)) / (d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, cached_v)

        # 5. Return rounded output and updated cache
        return torch.round(output, decimals=4), kv_cache