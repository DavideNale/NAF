import torch
import torch.nn as nn
import numpy as np
from modules import find_features

# Neural Acoustic Field Network
class NAF(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1,
        grid_density=0.15, feature_dim=64, min_xy=None, max_xy=None):
        super(NAF, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(qualocosa, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), nn.ReLU(),
        )

        self.skip = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        
        # Create grid
        grid_coords_x = np.arange(min_xy[0], max_xy[0], grid_density)
        grid_coords_y = np.arange(min_xy[1], max_xy[1], grid_density)
        grid_coords_x, grid_coords_y = np.meshgrid(grid_coords_x, grid_coords_y)
        grid_coords_x = grid_coords_x.flatten()
        grid_coords_y = grid_coords_y.flatten()
        xy_train = np.array([grid_coords_x, grid_coords_y]).T

        self.register_buffer("grid_coords_xy", torch.from_numpy(xy_train).float(), persistent=True)
        self.xy_offset = nn.Parameter(torch.zeros_like(self.grid_coords_xy), requires_grad=True)
        self.grid_0 = nn.Parameter(torch.randn(len(grid_coords_x), feature_dim, device="cpu").float() / np.sqrt(float(feature_dim)), requires_grad=True)

        # Initalization of layers via custom moduels
        self.first = intermediate(input_dim, hidden_dim)
        self.layers = torch.nn.ModuleList()
        for k in range(6):
            self.layers.append(intermediate(hidden_dim, hidden_dim))
        self.skip = skip(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, input, srcs, mics):
        SAMPLES = input.shape[1]
        srcs = srcs[..., :2]
        mics = mics[..., :2]

        features_srcs = find_features(srcs, self.grid_coords_xy, self.grid_0).expand(-1, SAMPLES, -1)
        features_mics = find_features(mics, self.grid_coords_xy, self.grid_0).expand(-1, SAMPLES, -1)

        total_input = torch.cat((features_srcs, features_mics, input), dim=2)

        # Passing throught the layers
        out = self.first(total_input)
        out = self.layers[0](out)
        out = self.layers[1](out)
        out = self.layers[2](out)

        skip = self.skip(total_input)
        
        out = self.layers[3](out + skip)
        out = self.layers[4](out)
        out = self.layers[5](out)
        out = self.output(out)

        return out