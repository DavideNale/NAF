import numpy as np
import torch
import torch.optim as optim
import math
from torch.utils.data import DataLoader
from network import NAF
from sound_loader import sound_samples
from modules import embedding_module_log
torch.manual_seed(42)

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Network parameters
feature_dim = 64
hidden_dim = 256
output_dim = 100

# Trainig parameters
learning_rate = 0.01
num_epochs = 200
batch_size = 20
ft_num = 200

# Dataset
print('Loading dataset. It might take a while ...')
dataset = sound_samples(num_samples=ft_num)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Spawn embedders and move to GPU
xyz_embedder = embedding_module_log(num_freqs=10, ch_dim=2, max_freq=7).to(device)
freq_embedder = embedding_module_log(num_freqs=10, ch_dim=2).to(device)
time_embedder = embedding_module_log(num_freqs=10, ch_dim=2).to(device)

# Spawn network and move to GPU
net = NAF(input_dim = 296, min_xy=dataset.min_pos[:2], max_xy=dataset.max_pos[:2]).to(device)
criterion = torch.nn.MSELoss()

# Create pools for optimization
orig_container = []
grid_container = []

for par_name, par_val in net.named_parameters():
    if "grid" in par_name:
        grid_container.append(par_val)
    else:
        orig_container.append(par_val)

optimizer = torch.optim.AdamW([
    {'params': grid_container, 'lr': learning_rate, 'weight_decay': 1e-2},
    {'params': orig_container, 'lr': learning_rate, 'weight_decay': 0}],
    lr = learning_rate,
    weight_decay = 0.0
)

# Create the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

print("Training started ...")

# Training loop
for epoch in range(num_epochs):
    for batch_index, batch in enumerate(dataloader):
        #print('Epoch: ', epoch, ' Batch: ', batch_index)

        # Unpack arrays and move to GPU
        gts = batch[0].to(torch.float32).to(device, non_blocking=True)
        srcs = batch[1].to(torch.float32).to(device, non_blocking=True).unsqueeze(1)
        mics = batch[2].to(torch.float32).to(device, non_blocking=True).unsqueeze(1)
        freqs = batch[3].to(torch.float32).to(device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi
        times = batch[4].to(torch.float32).to(device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi

        # Embeddings
        with torch.no_grad():
            src_embed = xyz_embedder(srcs).expand(-1, ft_num, -1)
            mic_embed = xyz_embedder(mics).expand(-1, ft_num, -1)
            freq_embed = freq_embedder(freqs)
            time_embed = time_embedder(times)

        # Concatenate and feed to network
        input = torch.concatenate((src_embed, mic_embed, freq_embed, time_embed), dim=2)
        optimizer.zero_grad(set_to_none=False)

        input = input.to(device)

        output = net(input, srcs, mics)
        output = np.squeeze(output, axis=-1)
        loss = criterion(output, gts)

        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"Epoch : {epoch}, loss : {loss}") 
