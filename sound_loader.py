import torch
import numpy as np
from pathlib import Path
import utils.irutilities as irutil
import json

class sound_samples(torch.utils.data.Dataset):
    def __init__(self, num_samples):
        # Args
        self.num_samples = num_samples
        
        # Load dataset and save spectrogram shape
        path = Path('mesh_rir/S32-M441_npy/')
        self.spectrograms = np.load(path.joinpath('spectrograms.npy'), mmap_mode='r+')
        self.posMic, self.posSrc, _ = irutil.loadIR(path)

        # Loading data metrics
        with open(path/'metrics.json', 'r') as json_file:
            loaded_data = json.load(json_file)

        self.mean = np.float32(loaded_data['mean'])
        self.std = np.float32(loaded_data['std'])
        self.min_value = np.float32(loaded_data["min"])
        self.max_value = np.float32(loaded_data["max"])

        # Calculate min_xy
        min_x = min(np.min(self.posMic[:,0]), np.min(self.posSrc[:,0]))
        min_y = min(np.min(self.posMic[:,1]), np.min(self.posSrc[:,1]))
        min_z = min(np.min(self.posMic[:,2]), np.min(self.posSrc[:,2]))
        self.min_pos = np.array((min_x, min_y, min_z))

        # Calculate max_xy
        max_x = max(np.max(self.posMic[:,0]), np.max(self.posSrc[:,0]))
        max_y = max(np.max(self.posMic[:,1]), np.max(self.posSrc[:,1]))
        max_z = max(np.max(self.posMic[:,2]), np.max(self.posSrc[:,2]))
        self.max_pos = np.array((max_x, max_y, max_z))

        # Join in a single list of (src, mic, IR) objects
        self.indices = []
        for s in range(len(self.posSrc)):
            for m in range(len(self.posMic)):
                self.indices.append([s,m])
                
    def __len__(self):
        return  len(self.indices)

    def __getitem__(self, idx):
        # Retrieve source and microphone position
        s, m = self.indices[idx]
        src = self.posSrc[s]
        mic = self.posMic[m]

        # Normalization -1:1
        src_norm = ((src - self.min_pos)/(self.max_pos-self.min_pos) - 0.5) * 2.0 + np.random.normal(0,1) * 0.01
        mic_norm = ((mic - self.min_pos)/(self.max_pos-self.min_pos) - 0.5) * 2.0 + np.random.normal(0,1) * 0.01

        # Sample <num_samples> frequencies and times from the spectrogram
        spectrogram = self.spectrograms[s,m]
        sound_size = spectrogram.shape
        freqs = np.random.randint(0, sound_size[0], self.num_samples)
        times = np.random.randint(0, sound_size[1], self.num_samples)

        # Normalization -1:1
        freqs_norm = (torch.tensor(freqs, dtype=torch.float32)/sound_size[0] - 0.5) * 2
        times_norm = (torch.tensor(times, dtype=torch.float32)/sound_size[1] - 0.5) * 2

        # Ground truths
        gts = spectrogram[freqs, times]

        return gts, src_norm, mic_norm, freqs_norm, times_norm

    