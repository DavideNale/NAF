import torch
import numpy as np
from pathlib import Path
import utils.irutilities as irutil

np.random.seed(0)
torch.manual_seed(0)

class sound_samples(torch.utils.data.Dataset):
    def __init__(self):
        # Load dataset
        path = Path('mesh_rir/S32-M441_npy/')
        self.spectrograms = np.load(path.joinpath('spectrograms.npy'))
        posMic, posSrc, _ = irutil.loadIR(path)

        # Join in a single list of (src, mic, IR) objects
        self.indices = []

        for s in posSrc:
            for m in posMic:
                self.indices.append([s,m])
                
    def __len__(self):
        return  len(self.indices)

    def __getitem__(self, idx):
        s, m = self.indices[idx]
        src = posSrc[s]
        mic = posMic[m]
        spectrogram = self.spectrograms[s,m]
        return src, mic, spectrogram
