import torch
import torch.nn as nn
import numpy as np
from modules import find_features

# A projection block that maps the input tensor to an output tensor
# via a linear trasformation with bias: Y=AX+b
class projection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(projection, self).__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=True)
    def forward(self, x):
        return self.proj(x)

# A sequential block that combines projection and leakyReLU
class sequential(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(sequential, self).__init__()
        self.block = nn.Sequential(nn.LeakyReLU(negative_slope=0.1), projection(input_dim, output_dim))
    def forward(self, x):
        return self.block(x)

# Neural Acoustic Field Network
class NAF(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=1, num_layers=8,
        grid_density=0.15, feature_dim=64, min_xy=None, max_xy=None):
        super(NAF, self).__init__()

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
        #self.bandwidth_min = bandwidth_min
        #self.bandwidth_max = bandwidth_max
        #self.float_amt = float_amt
        #self.bandwidths = nn.Parameter(torch.zeros(len(grid_coors_x))+grid_bandwidth, requires_grad=True)
        #self.register_buffer("grid_coors_xy",torch.from_numpy(xy_train).float(), persistent=True)
        #self.xy_offset = nn.Parameter(torch.zeros_like(self.grid_coors_xy), requires_grad=True)
        #self.grid_0 = nn.Parameter(torch.randn(len(grid_coors_x),grid_ch, device="cpu").float() / np.sqrt(float(grid_ch)), requires_grad=True)



        # Initalization of custom moduels
        self.proj = projection(input_dim + 2*feature_dim, hidden_dim)
        self.residual = nn.Sequential(projection(input_dim + 2*feature_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.1))
        
        # Define layers
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.layers.append(sequential(hidden_dim, output_dim))
                
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, srcs, mics):
        srcs_xy = srcs[..., :2]
        mics_xy = mics[..., :2]

        
        features_srcs = torch.tensor(find_features(srcs_xy, self.grid_coords_xy, self.grid_0)).to('cuda')
        features_mics = torch.tensor(find_features(mics_xy, self.grid_coords_xy, self.grid_0)).to('cuda')
        print(features_srcs.shape, features_mics.shape, input.shape)

        total_input = torch.cat((features_mics, features_mics, input), dim=2)
        print(total_input.shape)

        out = self.proj(input)
        # Pass throught all layers of the network
        for k in range(len(self.layers)):
            out = self.layers[k](out)
            if k == (self.blocks // 2 - 1):
                out = out + self.residual(total_input).unsqueeze(2).repeat(1,1,2,1)

        return out
        