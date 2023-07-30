import torch
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt
import subprocess
import argparse
import os
import tempfile

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

s = 12
m = 30
sample = dataset.spectrograms[s,m,:]
src = dataset.posSrc[s, :]
mic = dataset.posMic[m, :]

src_norm = ((src - dataset.min_pos)/(dataset.max_pos-dataset.min_pos) - 0.5) * 2.0
mic_norm = ((mic - dataset.min_pos)/(dataset.max_pos-dataset.min_pos) - 0.5) * 2.0

src_norm = torch.tensor(src_norm).to(device).float()
mic_norm = torch.tensor(mic_norm).to(device).float()

# Spawn embedders and move to GPU
xyz_embedder = embedding_module_log(num_freqs=7, ch_dim=0, max_freq=7).to(device)
freq_embedder = embedding_module_log(num_freqs=7, ch_dim=1).to(device)
time_embedder = embedding_module_log(num_freqs=7, ch_dim=1).to(device)

# Load NAF with selected configuration
net = NAF(input_dim = 248, min_xy=dataset.min_pos[:2], max_xy=dataset.max_pos[:2]).to(device)
state_dict = torch.load(args.config_file)
net.load_state_dict(state_dict)
net.eval()

test = np.ones((1025,65,248))

i = 0   

# Run in inference
with torch.no_grad():

    src_embed = xyz_embedder(src_norm).unsqueeze(0).unsqueeze(1).repeat(1025, 65, 1)
    mic_embed = xyz_embedder(mic_norm).unsqueeze(0).unsqueeze(1).repeat(1025, 65, 1)
    
    for f in range(1025):
        for t in range(65):
            test[f,t]=i
            i=i+1
            
#    print(src_embed)
#    print(mic_embed)
#    f = torch.arange(1025).unsqueeze(1).to(device)/1024
#    print(f)
#    t = torch.arange(65).unsqueeze(1).to(device)/64
#    print(t)
#
#    freq_embed = freq_embedder(f).unsqueeze(1).repeat(1,65,1)
#    print(freq_embed,freq_embed.shape)
#    time_embed = time_embedder(t).unsqueeze(0).repeat(1025,1,1)
#    print(time_embed, time_embed.shape)
#
#    input = torch.concatenate((src_embed, mic_embed, freq_embed, time_embed), dim=2)
#    input = input.to(device).float()
#
#    output = net(input, src_norm.unsqueeze(0).unsqueeze(1).repeat(1025,1,3), mic_norm.unsqueeze(0).unsqueeze(1).repeat(1025,1,3))
#    output = output.cpu()
#    output = (output * dataset.std_deviation)+dataset.mean_value

# Generate and save image to temp directory
plt.imshow(test, cmap='hot', aspect='auto')
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram Heatmap')

temp_image = os.path.join(tempfile.gettempdir(), 'temp_heatmap.png')
plt.savefig(temp_image, bbox_inches='tight')
plt.close()

subprocess.run("imv "+temp_image, shell=True, check=True, text=True, capture_output=True)