import numpy as np
import librosa
import os
from pathlib import Path

import utils.irutilities as irutil

# dataset path
path = Path('mesh_rir/S32-M441_npy/')

print('Loading IRs ...')
posMic, posSrc, ir = irutil.loadIR(path)
n_fft = 2048  # Number of FFT points
hop_length = 512  # Hop length (stride) between consecutive frames
ir_shape = np.abs(librosa.stft(ir[0,0,:], n_fft=n_fft, hop_length=hop_length)).shape
print('Spectrogram shape: ', ir_shape)

spectrograms = np.zeros((32, 441, ir_shape[0], ir_shape[1]))

print('Computing spectrograms ...')

for m in range(posMic.shape[0]):
    for s in range(posSrc.shape[0]):
        sample = ir[s,m,:]

        spectrogram = np.abs(librosa.stft(sample, n_fft=n_fft, hop_length=hop_length))
        log_mag_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

        spectrograms[s, m] = log_mag_spectrogram

print('Done:')
memory_size = spectrograms.nbytes / (1024 ** 2)
print("Memory size:", memory_size, "MB")
print("Array shape: ", spectrograms.shape)

# Save the spectrograms
save_path = path / 'spectrograms.npy'
np.save(save_path, spectrograms)

print(f'Spectrograms saved to: {save_path}')