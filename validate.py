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
src_norm = torch.tensor(src).mul_(math.pi).to(device).float()
mic_norm = torch.tensor(mic).mul(math.pi).to(device).float()

# Spawn embedders and move to GPU
xyz_embedder = embedding_module_log(num_freqs=7, ch_dim=0, max_freq=7).to(device)
freq_embedder = embedding_module_log(num_freqs=7, ch_dim=0).to(device)
time_embedder = embedding_module_log(num_freqs=7, ch_dim=0).to(device)

# Load NAF with selected configuration
net = NAF(input_dim = 248, min_xy=dataset.min_pos[:2], max_xy=dataset.max_pos[:2]).to(device)
state_dict = torch.load(args.config_file)
net.load_state_dict(state_dict)
net.eval()


input = torch.zeros((1025,65)).to(device)

# Run in inference
with torch.no_grad():

    src_embed = xyz_embedder(src_norm)
    mic_embed = xyz_embedder(mic_norm)

    for f in range(1025):
        for t in range(65):
            f_emb = freq_embedder(torch.tensor(f).unsqueeze(0).to(device))
            t_emb = time_embedder(torch.tensor(t).unsqueeze(0).to(device))

            input[f,t] = net.test(sample[f,t])


    # src_embed = xyz_embedder(src_norm)
    # mic_embed = xyz_embedder(mic_norm)

    # f_range = torch.arange(1025)
    # t_range = torch.arange(65)

    # combinations = torch.cartesian_prod(f_range, t_range)
    # results = []

    # for i, (f,t) in enumerate(combinations):
    #     f_emb = freq_embedder(torch.tensor(f).unsqueeze(0).to(device))
    #     t_emb = time_embedder(torch.tensor(t).unsqueeze(0).to(device))
    #     line = torch.cat((src_embed, mic_embed,f_emb, t_emb), dim=0)
    #     results.append(line)

    # input = torch.vstack(results)

    # output = net(input, src_norm.unsqueeze(0).unsqueeze(1).repeat(1025,1,3), mic_norm.unsqueeze(0).unsqueeze(1).repeat(1025,1,3))
    # output = output.cpu()
    # output.squeeze_(2)
    # print(output.shape)   

# # Generate and save image to temp directory
# plt.imshow(output, cmap='hot', aspect='auto')
# plt.colorbar()
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.title('Spectrogram Heatmap')

# First Image
plt.subplot(1, 2, 1)
plt.imshow(sample, cmap='hot', aspect='auto')
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram 1')

# Second Image
plt.subplot(1, 2, 2)
plt.imshow(input.cpu(), cmap='hot', aspect='auto')
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram 2')


temp_image = os.path.join(tempfile.gettempdir(), 'temp_heatmap.png')
plt.savefig(temp_image, bbox_inches='tight')
plt.close()

subprocess.run("imv "+temp_image, shell=True, check=True, text=True, capture_output=True)