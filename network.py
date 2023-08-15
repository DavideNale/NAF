import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import cKDTree

from modules import find_features, positional_encoding

# Neural Acoustic Field Network
class NAF(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=1,
        embedding_dim_pos=7, embedding_dim_spectro=10,
        grid_density=0.15, feature_dim=64, min_xy=None, max_xy=None):
        super(NAF, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.1),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, output_dim), nn.LeakyReLU(negative_slope=0.1),
        )
        self.skip = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.1),
        )

        # Positional encoding
        self.xyz_encoder = positional_encoding()
        self.ft_encoder = positional_encoding(ch_dim=2)
        
        # Create grid
        grid_coords_x = np.arange(min_xy[0], max_xy[0], grid_density)
        grid_coords_y = np.arange(min_xy[1], max_xy[1], grid_density)
        grid_coords_x, grid_coords_y = np.meshgrid(grid_coords_x, grid_coords_y)
        grid_coords_x = grid_coords_x.flatten()
        grid_coords_y = grid_coords_y.flatten()
        xy_train = np.array([grid_coords_x, grid_coords_y]).T

        self.kdtree = cKDTree(xy_train)

        self.register_buffer("grid_coords_xy", torch.from_numpy(xy_train).float(), persistent=True)
        self.xy_offset = nn.Parameter(torch.zeros_like(self.grid_coords_xy), requires_grad=True)
        self.grid_0 = nn.Parameter(torch.randn(len(grid_coords_x), feature_dim, device="cpu").float() / np.sqrt(float(feature_dim)), requires_grad=True)

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, srcs, mics, freqs, times):
        SAMPLES = freqs.shape[1]
        srcs_xy = srcs[:,:2].cpu()
        mics_xy = mics[:,:2].cpu()
        pos = torch.cat((srcs, mics), dim=1)
        pos_enc = self.xyz_encoder(pos).unsqueeze(1).expand(-1, SAMPLES, -1)

        freqs = freqs.unsqueeze(2)
        times = times.unsqueeze(2)
        freqs_enc = self.ft_encoder(freqs)
        times_enc = self.ft_encoder(times)


        # Exctract srcs features from the grid
        _, indices = self.kdtree.query(srcs_xy, k=1)
        features_srcs = torch.vstack([self.grid_0[indices]]).unsqueeze(1).expand(-1, SAMPLES, -1)

        # Exctract mics features from the grid
        distances, indices = self.kdtree.query(mics_xy, k=1)
        features_mics = torch.vstack([self.grid_0[indices]]).unsqueeze(1).expand(-1, SAMPLES, -1)

        input = torch.cat((features_srcs, features_mics, pos_enc, freqs_enc, times_enc), dim=2)

        # Passing throught the layers
        out = self.block1(input)
        temp = self.skip(input)
        out = self.block2(temp+out)

        return out

    def spectrogram_at(self, src, mic):
        with torch.no_grad():
            src = src.t().to('cuda').float()
            mic = mic.t().to('cuda').float()

            # f,t encoding
            f_range = torch.arange(1025).to('cuda').unsqueeze(1)
            t_range = torch.arange(65).to('cuda').unsqueeze(0)
            fs = f_range.expand(-1, 65).flatten().unsqueeze(0)
            ts = t_range.expand(1025, -1).flatten().unsqueeze(0)

            out = self.forward(src, mic, fs, ts)
            out = torch.reshape(out, (1025,65)) 
        return out