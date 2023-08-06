import numpy as np
import torch
import torch.optim as optim
import math
from torch.utils.data import DataLoader
import datetime

from network import NAF
from sound_loader import sound_samples
from modules import embedding_module_log

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Trainig parameters
learning_rate = 0.0005
decay_rate = 0.05
num_epochs = 50
batch_size = 20
ft_num = 200

# Dataset
print('Loading dataset. It might take a while ...')
dataset = sound_samples(num_samples=ft_num)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

# Spawn embedders and move to GPU
xyz_embedder = embedding_module_log(num_freqs=4, ch_dim=2, max_freq=7).to(device)
freq_embedder = embedding_module_log(num_freqs=10, ch_dim=2).to(device)
time_embedder = embedding_module_log(num_freqs=10, ch_dim=2).to(device)

# Spawn network and move to GPU
net = NAF(input_dim = 248, min_xy=dataset.min_pos[:2], max_xy=dataset.max_pos[:2]).to(device)
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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=7, factor=0.5, verbose=True)

print("Training started ...")
average_loss=0

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0

    for batch_index, batch in enumerate(dataloader):
        # Unpack arrays and move to GPU
        gts = batch[0].to(device, dtype=torch.float32)
        gts = gts / dataset.max_amplitude  # Normalization [0,1]
        srcs = batch[1].to(device, dtype=torch.float32).unsqueeze(1).mul(math.pi)
        mics = batch[2].to(device, dtype=torch.float32).unsqueeze(1).mul(math.pi)
        freqs = batch[3].to(device, dtype=torch.float32).unsqueeze(2)
        times = batch[4].to(device, dtype=torch.float32).unsqueeze(2)

        # Embeddings
        with torch.no_grad():
            src_embed = xyz_embedder(srcs).expand(-1, ft_num, -1)
            mic_embed = xyz_embedder(mics).expand(-1, ft_num, -1)
            freq_embed = freq_embedder(freqs)
            time_embed = time_embedder(times)

        # Concatenate and feed to network
        input = torch.cat((src_embed, mic_embed, freq_embed, time_embed), dim=2)

        output = net(input, srcs, mics)
        output = torch.squeeze(output, dim=-1)
        loss = criterion(output, gts)

        optimizer.zero_grad()
        loss.backward()
        running_loss += loss.item()
        optimizer.step()

    # Calculating the new learning rate
    # new_lr = learning_rate * (decay_rate ** (epoch + 1 / num_epochs))
    # print(new_lr)

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = new_lr
    # Average loss for the epoch
    average_loss = running_loss / len(dataloader)
    print(f"Epoch : {epoch}, loss : {average_loss}") 

    # Step the learning rate scheduler after each epoch
    scheduler.step(average_loss)
    
# Save the model configuration after training is complete
print("Saving configuration")
current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'net_{current_date}_loss_{average_loss:.4f}.pth'
torch.save(net.state_dict(), 'saved/' + filename)
