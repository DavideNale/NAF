import numpy as np
import torch
import torch.optim as optim
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
learning_rate = 0.001
num_epochs = 200
batch_size = 20
ft_num = 2000

# Dataset
print('Loading dataset. It might take a while ...')
dataset = sound_samples(num_samples=2000)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
print('min: ', dataset.min_xy)
print('max: ', dataset.max_xy)

# Spawn embedders and move to GPU
xyz_embedder = embedding_module_log(num_freqs=10).to(device)
freq_embedder = embedding_module_log(num_freqs=10).to(device)
time_embedder = embedding_module_log(num_freqs=10).to(device)

# Spawn network and move to GPU
net = NAF(input_dim = 296, min_xy=dataset.min_xy, max_xy=dataset.max_xy).to(device)
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

# Training loop
for epoch in range(num_epochs):
    for batch_index, batch in enumerate(dataloader):
        print('Epoch: ', epoch, ' Batch: ', batch_index)

        # Unpack arrays and move to GPU
        gts = batch[0].float().to(device)
        srcs = batch[1].to(device)
        mics = batch[2].to(device)
        freqs = batch[3].to(device)
        times = batch[4].to(device)

        # Embeddings
        with torch.no_grad():
            src_embed = xyz_embedder(srcs).reshape(srcs.shape[0], 2000, -1)
            mic_embed = xyz_embedder(mics).reshape(mics.shape[0], 2000, -1)
            freq_embed = freq_embedder(freqs).reshape(freqs.shape[0], 2000, -1)
            time_embed = time_embedder(times).reshape(times.shape[0], 2000, -1)

        # Concatenate and feed to network
        input = torch.concatenate((src_embed, mic_embed, freq_embed, time_embed), dim=2)
        optimizer.zero_grad(set_to_none=False)
        
        output = net(input.to(torch.float32), srcs, mics)
        output = np.squeeze(output, axis=-1)
        loss = criterion(output, gts)
        loss.backward()
        optimizer.step()

    print(f"Epoch : {epoch}, loss : {loss}") 
