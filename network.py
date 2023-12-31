import torch
import librosa
import numpy as np
import torch.nn as nn
from scipy.spatial import cKDTree


class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=10, ch_dim=1):
        super().__init__()
        self.funcs = [torch.sin, torch.cos]
        self.num_functions = len(self.funcs)
        self.freqs = (2 ** torch.arange(0, num_freqs)) * torch.pi
        self.ch_dim = ch_dim

    def forward(self, x_input):
        # out_list = [func(x_input * self.freqs.view(-1, 1)) for func in self.funcs]
        # return torch.cat(out_list, dim=self.ch_dim)
        out_list = []
        for func in self.funcs:
            for freq in self.freqs:
                out_list.append(func(x_input * freq))
        return torch.cat(out_list, dim=self.ch_dim)


class NAF(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=256, tail_spectro_dim=128, tail_phase_dim=128,
        output_dim=2, embedding_dim_pos=7, embedding_dim_spectro=10, grid_density=0.10,
        feature_dim=64, min_xy=None, max_xy=None
        ):
        super(NAF, self).__init__()

        self.feature_dim = feature_dim

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.1),
        )

        self.tail1 = nn.Sequential(
            nn.Linear(hidden_dim, tail_spectro_dim), nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(tail_spectro_dim, tail_spectro_dim), nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(tail_spectro_dim, tail_spectro_dim), nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(tail_spectro_dim, 1),
        )

        self.tail2 = nn.Sequential(
            nn.Linear(hidden_dim, tail_phase_dim), nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(tail_phase_dim, tail_phase_dim), nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(tail_phase_dim, tail_phase_dim), nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(tail_phase_dim, 1),
        )

        # Positional encoding
        self.xyz_encoder = PositionalEncoding()
        self.ft_encoder = PositionalEncoding(ch_dim=2)

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
        self.grid_0 = nn.Parameter(
            torch.randn(len(grid_coords_x), feature_dim, device="cpu").float()
            / np.sqrt(float(feature_dim)),
            requires_grad=True,
        )

    def get_grid(self):
        return self.grid_0.clone()

    def forward(self, srcs, mics, freqs, times):
        SAMPLES = freqs.shape[1]
        srcs_xy = srcs[:, :2].cpu()
        mics_xy = mics[:, :2].cpu()
        pos = torch.cat((srcs, mics), dim=1)
        pos = pos + torch.rand(pos.shape).to("cuda") * 0.05
        pos_enc = self.xyz_encoder(pos).unsqueeze(1).expand(-1, SAMPLES, -1)

        freqs = freqs.unsqueeze(2)
        times = times.unsqueeze(2)
        freqs_enc = self.ft_encoder(freqs)
        times_enc = self.ft_encoder(times)

        # # Exctract srcs features from the grid
        distances, indices = self.kdtree.query(srcs_xy, k=1)
        features_srcs = (
            torch.vstack([self.grid_0[indices]]).unsqueeze(1).expand(-1, SAMPLES, -1)
        )

        # # Exctract mics features from the grid
        distances, indices = self.kdtree.query(mics_xy, k=1)
        features_mics = (
            torch.vstack([self.grid_0[indices]]).unsqueeze(1).expand(-1, SAMPLES, -1)
        )
        input = torch.cat(
            (features_srcs, features_mics, pos_enc, freqs_enc, times_enc), dim=2
        )

        # Passing throught the layers
        out = self.block1(input)
        # temp = self.skip(input)
        # out = out + temp
        tail1 = self.tail1(out)
        tail2 = self.tail2(out)
        # out = self.block2(out+temp)
        out = torch.cat((tail1, tail2), dim=-1)

        return out

    def spectrogram_at(self, src, mic):
        with torch.no_grad():
            src = src.t().to("cuda").float()
            mic = mic.t().to("cuda").float()

            # f,t encoding
            f_range = torch.arange(257).to("cuda").unsqueeze(1) / 257
            f_range = (f_range - 0.5) * 2
            t_range = torch.arange(257).to("cuda").unsqueeze(0) / 257
            t_range = (t_range - 0.5) * 2

            fs = f_range.expand(-1, 257).flatten().unsqueeze(0)
            ts = t_range.expand(257, -1).flatten().unsqueeze(0)

            out = self.forward(src, mic, fs, ts)
            out = -torch.reshape(out, (257, 257, 2))
        return out

    def audio_at(self, src, mic):
        out = self.spectrogram_at(src, mic).cpu().numpy()
        s = out[:, :, 0] * 40
        p = out[:, :, 1] * np.pi * 180
        return to_audio(s, p)

    def loudness_map(self, src, resolution):
        src = torch.tensor(src).unsqueeze(1).to("cuda")

        x_values = torch.linspace(-0.5, 0.5, resolution)
        y_values = torch.linspace(-0.5, 0.5, resolution)
        xx, yy = torch.meshgrid(x_values, y_values, indexing="ij")
        grid_points = torch.vstack((xx.ravel(), yy.ravel())).t()

        ones = torch.ones((grid_points.shape[0], 1))
        grid_points = torch.hstack((grid_points, ones))

        result = np.zeros(grid_points.shape[0])

        for i, point in enumerate(grid_points):
            mic = torch.tensor(point).unsqueeze(1).to("cuda")
            out = self.spectrogram_at(src, mic)

            out_s = (out[:, :, 0] * 40) - 40
            out_p = out[:, :, 1] * np.pi * 180

            out_s = librosa.db_to_amplitude(out_s.cpu()) * 135

            audio = librosa.istft(
                out_s.numpy() * np.exp(1j * out_p.cpu().numpy()), hop_length=512 // 4
            )
            peak = np.max(audio)
            result[i] = peak

        result = result.reshape(resolution, resolution)
        return result
