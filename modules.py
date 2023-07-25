import torch
from torch import nn
from torch import pairwise_distance
import numpy as np
import math

class embedding_module_log(nn.Module):
    def __init__(self, funcs=[torch.sin, torch.cos], num_freqs=20, max_freq=10, ch_dim=1, include_in=True):
        super().__init__()
        self.functions = funcs
        self.num_functions = list(range(len(funcs)))
        self.freqs = torch.nn.Parameter(2.0**torch.from_numpy(np.linspace(start=0.0,stop=max_freq, num=num_freqs).astype(np.single)), requires_grad=False)
        self.ch_dim = ch_dim
        self.funcs = funcs
        self.include_in = include_in

    def forward(self, x_input):
        if self.include_in:
            out_list = [x_input]
        else:
            out_list = []
        for func in self.funcs:
            for freq in self.freqs:
                out_list.append(func(x_input*freq))
        return torch.cat(out_list, dim=self.ch_dim)

# Pairwise euclidean distance
def euclidean_distance(x1:torch.Tensor, x2:torch.Tensor) -> torch.Tensor:
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    return torch.addmm(x2_norm.transpose(-2,1), x1, x2.transpose(-2,-1), alpha=-2).add_(x1_norm)

#@torch.jit.script
def find_features(input_pos:torch.Tensor, grid_pos:torch.Tensor, grid:torch.Tensor) -> torch.Tensor:
    distance = euclidean_distance(input_pos.squeeze(1), grid_pos)

    min_indices = torch.argmin(distance, dim=-1)
    features = grid[min_indices,:]
    return features
