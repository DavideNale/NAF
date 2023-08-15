import numpy as np
import librosa
import os
from pathlib import Path
import json
import torch
import gc

import utils.irutilities as irutil

# Load IRs
path = Path('mesh_rir/S32-M441_npy/')
print('Loading IRs ...')
posMic, posSrc, ir = irutil.loadIR(path)

H = 1025
W = 65

spectrograms = np.memmap(
    'spectrograms.temp', 
    dtype=np.float32,
    mode='w+',
    shape = (32,441,1025,65)
)

# Remember to flush and close the memory-mapped array when done
# del mmapped_array

print('Computing spectrograms ...')

for m in range(posMic.shape[0]):
    for s in range(posSrc.shape[0]):
        sample = ir[s,m,:]

        spectrogram = np.abs(librosa.stft(sample, n_fft=2048, hop_length=512))
        log_mag_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

        spectrograms[s, m] = log_mag_spectrogram


gc.collect()
print('Calculating metrics')

mean = np.mean(spectrograms)
std = np.std(spectrograms)
min_val = np.min(spectrograms)
max_val = np.max(spectrograms)

spectrograms = (spectrograms - mean) / std

#min_value = np.min(spectrograms)
#max_value = np.max(spectrograms)
data = {
    "mean": float(mean),
    "std": float(std),
    "min": float(min_val),
    "max": float(max_val)
}

print('Done:')
memory_size = spectrograms.nbytes / (1024 ** 2)
print("Memory size:", memory_size, "MB")
print("Array shape: ", spectrograms.shape)

with open(path/'metrics.json', 'w') as json_file:
    json.dump(data, json_file)

# Save the spectrograms
save_path = path / 'spectrograms.npy'
np.save(save_path, spectrograms)

del spectrograms
print(f'Spectrograms saved to: {save_path}')