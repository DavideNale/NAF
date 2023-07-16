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

def euclidean_distance(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    distance = math.sqrt(dx ** 2 + dy ** 2)
    return distance

def find_features(positions, coordinates, features):
    unique = positions[:,0,:].cpu()
    coordinates = coordinates.cpu()
    features = features.cpu().detach().numpy()
    output = np.zeros((20, 1, features.shape[1]))
    for i in range(unique.shape[0]):
        xy = unique[i,:]
        coords = coordinates
        norm = np.linalg.norm(coords-xy,axis=1)
        index = np.argmin(norm)
        output[i,:,:] = features[index,:]
    return np.repeat(output, 2000, axis=1)

