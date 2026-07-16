import math
import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)

        # Order matters: Key, Query, Value
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Linear projections
        K = self.key(embedded)
        Q = self.query(embedded)
        V = self.value(embedded)

        # 2. Scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))

        # 3. Causal mask
        seq_len = embedded.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=embedded.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # 4. Softmax
        scores = torch.softmax(scores, dim=2)

        # 5. Weighted sum of values
        output = torch.matmul(scores, V)

        return torch.round(output, decimals=4)