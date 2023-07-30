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
num_epochs = 100
batch_size = 20
ft_num = 2000

# Dataset
print('Loading dataset. It might take a while ...')
dataset = sound_samples(num_samples=ft_num)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Spawn embedders and move to GPU
xyz_embedder = embedding_module_log(num_freqs=7, ch_dim=2, max_freq=7).to(device)
freq_embedder = embedding_module_log(num_freqs=7, ch_dim=2).to(device)
time_embedder = embedding_module_log(num_freqs=7, ch_dim=2).to(device)

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

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0

    for batch_index, batch in enumerate(dataloader):
        # Unpack arrays and move to GPU
        gts = batch[0].to(device, dtype=torch.float32)
        gts = -(gts - dataset.mean_value) / dataset.std_deviation  # Normalization
        srcs = batch[1].to(device, dtype=torch.float32).unsqueeze(1)
        mics = batch[2].to(device, dtype=torch.float32).unsqueeze(1)
        freqs = batch[3].to(device, dtype=torch.float32).unsqueeze(2) * 2.0 * math.pi
        times = batch[4].to(device, dtype=torch.float32).unsqueeze(2) * 2.0 * math.pi

        # Embeddings
        with torch.no_grad():
            src_embed = xyz_embedder(srcs).expand(-1, ft_num, -1)
            mic_embed = xyz_embedder(mics).expand(-1, ft_num, -1)
            freq_embed = freq_embedder(freqs)
            time_embed = time_embedder(times)

        # Concatenate and feed to network
        input = torch.cat((src_embed, mic_embed, freq_embed, time_embed), dim=2)
        optimizer.zero_grad()

        input = input.to(device)
        output = net(input, srcs, mics)
        output = torch.squeeze(output, dim=-1)
        loss = criterion(output, gts)

        loss.backward()
        running_loss += loss.item()
        
    # Average loss for the epoch
    average_loss = running_loss / len(dataloader)
    print(f"Epoch : {epoch}, loss : {average_loss}") 

    # Step the learning rate scheduler after each epoch
    scheduler.step(average_loss)
    
# # Training loop
# for epoch in range(num_epochs):
#     for batch_index, batch in enumerate(dataloader):

#         # Unpack arrays and move to GPU
#         gts = batch[0].to(torch.float32).to(device, non_blocking=True)
#         gts = -(gts - dataset.mean_value) / dataset.std_deviation # Normalization
#         srcs = batch[1].to(torch.float32).to(device, non_blocking=True).unsqueeze(1)
#         mics = batch[2].to(torch.float32).to(device, non_blocking=True).unsqueeze(1)
#         freqs = batch[3].to(torch.float32).to(device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi
#         times = batch[4].to(torch.float32).to(device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi

#         # Embeddings
#         with torch.no_grad():
#             src_embed = xyz_embedder(srcs).expand(-1, ft_num, -1)
#             mic_embed = xyz_embedder(mics).expand(-1, ft_num, -1)
#             freq_embed = freq_embedder(freqs)
#             time_embed = time_embedder(times)

#         # Concatenate and feed to network
#         input = torch.concatenate((src_embed, mic_embed, freq_embed, time_embed), dim=2)
#         optimizer.zero_grad(set_to_none=False)

#         input = input.to(device)
#         output = net(input, srcs, mics)
#         output = np.squeeze(output, axis=-1)
#         loss = criterion(output, gts)

#         loss.backward()
#         optimizer.step()
        
#     print(f"Epoch : {epoch}, loss : {loss}") 
#     scheduler.step(loss)

# Save the model configuration after training is complete
print("Saving configuration")
current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'net_{current_date}_loss_{loss:.4f}.pth'
torch.save(net.state_dict(), 'saved/' + filename)
