import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt
import subprocess
import argparse
import os
import tempfile
import math

from network import NAF
from modules import embedding_module_log
from sound_loader import sound_samples

# Arguments handling
parser = argparse.ArgumentParser()
parser.add_argument("config_file", type=str, help="Path to the model's configuration file")
args = parser.parse_args()

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset
print('Loading dataset. It might take a while ...')
dataset = sound_samples(num_samples=20)

# Get one sample to compare to
s = 1
m = 100
sample = dataset.spectrograms[s,m,:]
src = dataset.posSrc[s, :]
mic = dataset.posMic[m, :]

# Normalize positions
src_norm = ((src - dataset.min_pos)/(dataset.max_pos-dataset.min_pos) - 0.5) * 2.0
mic_norm = ((mic - dataset.min_pos)/(dataset.max_pos-dataset.min_pos) - 0.5) * 2.0
src_norm = torch.tensor(src_norm).to(device).float()
mic_norm = torch.tensor(mic_norm).to(device).float()

# Spawn embedders and move to GPU
xyz_embedder = embedding_module_log(num_freqs=7, ch_dim=0, max_freq=7).to(device)
freq_embedder = embedding_module_log(num_freqs=7, ch_dim=0).to(device)
time_embedder = embedding_module_log(num_freqs=7, ch_dim=0).to(device)

# Load NAF with selected configuration
net = NAF(input_dim = 248, min_xy=dataset.min_pos[:2], max_xy=dataset.max_pos[:2]).to(device)
state_dict = torch.load(args.config_file)
net.load_state_dict(state_dict)
net.eval()

# Run in inference
with torch.no_grad():

    src_embed = xyz_embedder(src_norm)
    mic_embed = xyz_embedder(mic_norm)

    f_range = torch.arange(1025).to(device)
    t_range = torch.arange(65).to(device)

    combinations = torch.cartesian_prod(f_range, t_range).to(device)
    results = []

    for i, (f,t) in enumerate(combinations):
        f = f.unsqueeze(0)
        t = t.unsqueeze(0)
        f_emb = freq_embedder(f)
        t_emb = time_embedder(t)
        line = torch.cat((src_embed, mic_embed,f_emb, t_emb), dim=0)
        results.append(line)

    input = torch.vstack(results)
    output = net(input.view(1025,-1,120), src_norm.unsqueeze(0).unsqueeze(1).repeat(1025,1,3), mic_norm.unsqueeze(0).unsqueeze(1).repeat(1025,1,3))
    output = output.cpu()
    output.squeeze_(2)
    print(output.shape)   

criterion = torch.nn.MSELoss()
print(sample.shape, output.shape)
loss = criterion(torch.tensor(sample).div_(dataset.max_amplitude), output)
print(loss)

# First Image
plt.subplot(1, 2, 1)
plt.imshow(sample, cmap='hot', aspect='auto')
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram 1')

# Second Image
plt.subplot(1, 2, 2)
plt.imshow(output, cmap='hot', aspect='auto')
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram 2')


temp_image = os.path.join(tempfile.gettempdir(), 'temp_heatmap.png')
plt.savefig(temp_image, bbox_inches='tight')
plt.close()

subprocess.run("imv "+temp_image, shell=True, check=True, text=True, capture_output=True)