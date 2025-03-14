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
import time

from network import NAF
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
src = torch.tensor(dataset.posSrc[s, :]).unsqueeze(1).to(device)
mic = torch.tensor(dataset.posMic[m, :]).unsqueeze(1).to(device)

# Load NAF with selected configuration
net = NAF(input_dim = 288, min_xy=dataset.min_pos[:2], max_xy=dataset.max_pos[:2]).to(device)
state_dict = torch.load(args.config_file)
net.load_state_dict(state_dict)
net.eval()

start_time = time.time()
out = net.spectrogram_at(src, mic)
exec_time = time.time() - start_time
print("generation took: ",exec_time)
out[:,:,0] = (out[:,:,0] * dataset.std) + dataset.mean

plt.figure(figsize=(8,10))
# First Image
plt.subplot(2, 2, 1)
plt.imshow(sample, cmap='hot', aspect='auto')
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram 1')

# Second Image
plt.subplot(2, 2, 2)
plt.imshow(out.cpu()[:,:,0], cmap='hot', aspect='auto')
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram 2')

# Fourth Image
plt.subplot(2, 2, 4)
plt.imshow(out.cpu()[:,:,1], cmap='hot', aspect='auto')
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram 2')


temp_image = os.path.join(tempfile.gettempdir(), 'temp_heatmap.png')
plt.savefig(temp_image, bbox_inches='tight')
plt.close()

subprocess.run("imv "+temp_image, shell=True, check=True, text=True, capture_output=True)