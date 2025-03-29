import torch

from torch import nn, einsum
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class EncoderDecoder(nn.Module):
    def __init__(self, dim, units, activation=nn.SiLU):
        super().__init__()
        self.nn = nn.ModuleList()
        for i, u in enumerate(units):
            self.nn.append(nn.Linear(dim, u))
            if i < len(units) - 1:
                self.nn.append(activation())
            dim = u
    
    def forward(self, x):
        for l in self.nn:
            x = l(x)
        return x