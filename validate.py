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
#parser.add_argument("save_image", type=bool, help="A flag that controls whether or not the image gets saved")
args = parser.parse_args()

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset
print('Loading dataset. It might take a while ...')
dataset = sound_samples(num_samples=20)

s = 24
m = 57
sample = dataset.spectrograms[s,m,:]
src = dataset.posSrc[s, :]
mic = dataset.posMic[m, :]

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

input = torch.zeros((1025,65,120)).to(device)

# Run in inference
with torch.no_grad():

    src_embed = xyz_embedder(src_norm)
    mic_embed = xyz_embedder(mic_norm)

    for f in range(1025):
        for t in range(65):
            f_emb = freq_embedder(torch.tensor(f).unsqueeze(0).to(device))
            t_emb = time_embedder(torch.tensor(t).unsqueeze(0).to(device))

            line = torch.cat((src_embed, mic_embed, f_emb, t_emb), dim=0)
            input[f,t,:] = line

    output = net(input, src_norm.unsqueeze(0).unsqueeze(1).repeat(1025,1,3), mic_norm.unsqueeze(0).unsqueeze(1).repeat(1025,1,3))
    output = output.cpu()
    output.squeeze_(2)    

# Generate and save image to temp directory
plt.imshow(output, cmap='hot', aspect='auto')
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram Heatmap')

temp_image = os.path.join(tempfile.gettempdir(), 'temp_heatmap.png')
plt.savefig(temp_image, bbox_inches='tight')
plt.close()

subprocess.run("imv "+temp_image, shell=True, check=True, text=True, capture_output=True)