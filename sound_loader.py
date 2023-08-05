import torch
import numpy as np
from pathlib import Path
import utils.irutilities as irutil

class sound_samples(torch.utils.data.Dataset):
    def __init__(self, num_samples):
        # Args
        self.num_samples = num_samples
        
        # Load dataset and save spectrogram shape
        path = Path('mesh_rir/S32-M441_npy/')
        self.spectrograms = np.load(path.joinpath('spectrograms.npy'), mmap_mode='r+')
        self.posMic, self.posSrc, _ = irutil.loadIR(path)

        # Calculate mean and standard deviation of the dataset
        # self.mean_value = np.mean(self.spectrograms)
        # self.std_deviation = np.std(self.spectrograms)
        self.max_amplitude = np.min(self.spectrograms)

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

        # Normalization
        src_norm = ((src - self.min_pos)/(self.max_pos-self.min_pos) - 0.5) * 2.0
        mic_norm = ((mic - self.min_pos)/(self.max_pos-self.min_pos) - 0.5) * 2.0

        # Sample <num_samples> frequencies and times from the spectrogram
        spectrogram = self.spectrograms[s,m]
        sound_size = spectrogram.shape
        times = np.random.randint(0, sound_size[1], self.num_samples)
        freqs = np.random.randint(0, sound_size[0], self.num_samples)

        # Normalization
        freqs_norm = freqs/sound_size[0]
        times_norm = times/sound_size[1]

        # Ground truths
        gts = spectrogram[freqs, times]

        return gts, src_norm, mic_norm, freqs_norm, times_norm

    def test(self, idx):
        # Retrieve source and microphone position
        s, m = self.indices[idx]
        src = self.posSrc[s]
        mic = self.posMic[m]

        # Normalization
        src_norm = ((src - self.min_pos)/(self.max_pos-self.min_pos) - 0.5) * 2.0
        mic_norm = ((mic - self.min_pos)/(self.max_pos-self.min_pos) - 0.5) * 2.0

        # Sample <num_samples> frequencies and times from the spectrogram
        spectrogram = self.spectrograms[s,m]
        sound_size = spectrogram.shape
        times = np.random.randint(0, sound_size[1], self.num_samples)
        freqs = np.random.randint(0, sound_size[0], self.num_samples)

        # Normalization
        times_norm = times/sound_size[1]
        freqs_norm = freqs/sound_size[0]

        # Ground truths
        gts = spectrogram[freqs, times]

        return gts, src_norm, mic_norm, freqs_norm, times_norm, spectrogram
        
        
