import torch
from torch import nn
from torch import pairwise_distance
import numpy as np
import math

    # @staticmethod
    # def positional_encoding(x, L):
    #     out = [x]
    #     for j in range(L):
    #         out.append(torch.sin(2 ** j * x))
    #         out.append(torch.cos(2 ** j * x))
    #     return torch.cat(out, dim=1)

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        
    def forward(self, pred, truth, weights):
        squared_errors = torch.square(pred - truth)
        weighted_squared_errors = squared_errors * weights.unsqueeze(1)
        loss = torch.mean(weighted_squared_errors)
        return loss


class positional_encoding(nn.Module):
    def __init__(self, num_freqs=10, ch_dim=1):
        super().__init__()
        self.funcs=[torch.sin, torch.cos]
        self.num_functions = list(range(len(self.funcs)))
        self.freqs = (2 ** torch.arange(0, num_freqs)) * torch.pi
        self.ch_dim = ch_dim

    def forward(self, x_input):
        out_list = []
        for func in self.funcs:
            for freq in self.freqs:
                out_list.append(func(x_input*freq))
        return torch.cat(out_list, dim=self.ch_dim)

# Pairwise euclidean distance
def euclidean_distance(x1:torch.Tensor, x2:torch.Tensor) -> torch.Tensor:
    x1 = x1.repeat(x2.shape[0],1)
    x1_p = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_p = x2.pow(2).sum(dim=-1, keepdim=True)
    ac = torch.mul(x1[:,0], x2[:,0]).unsqueeze(1)
    bd = torch.mul(x1[:,1], x2[:,1]).unsqueeze(1)
    distances = x1_p + x2_p - 2*ac -2*bd
    index = torch.argmin(distances)
    return index

def find_features(input_pos:torch.Tensor, grid_pos:torch.Tensor, grid:torch.Tensor) -> torch.Tensor:
    indices = torch.vstack([euclidean_distance(row, grid_pos) for row in input_pos])
    return grid[indices,:]
