import numpy as np
import torch
import torch.optim as optim
import math
from torch.utils.data import DataLoader
import datetime
import wandb

from network import NAF
from sound_loader import sound_samples
from modules import positional_encoding

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Trainig parameters
learning_rate = 0.001
num_epochs = 300
batch_size = 20
ft_num = 2000

wandb.init(
    # set the wandb project where this run will be logged
    project="NAF",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "NAF",
    "dataset": "MeshRIR",
    "epochs": num_epochs,
    "batch_size": batch_size,
    "ft_num": ft_num,
    }
)

# Dataset
print('Loading dataset. It might take a while ...')
dataset = sound_samples(num_samples=ft_num)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)

# Spawn network and move to GPU
net = NAF(input_dim = 288, min_xy=dataset.min_pos[:2], max_xy=dataset.max_pos[:2]).to(device)
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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
#lr_lambda = lambda epoch: 0.99
#scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda)

print("Training started ...")
average_loss=0

grid_pre = net.get_grid().cpu().sum(dim=1)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0

    for batch_index, batch in enumerate(dataloader):
        # Unpack arrays and move to GPU
        gts = (batch[0].to(device, dtype=torch.float32) - dataset.mean) / dataset.std
        srcs = batch[1].to(device, dtype=torch.float32)
        mics = batch[2].to(device, dtype=torch.float32)
        freqs = batch[3].to(device, dtype=torch.float32)
        times = batch[4].to(device, dtype=torch.float32)

        # Feed to the network
        output = net(srcs, mics, freqs, times).squeeze(2)

        # Calculate loss
        # print(torch.max(torch.abs(output)))
        loss = criterion(gts, -output)

        # Perform backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    average_loss = (running_loss / len(dataloader))/batch_size
    print(f"Epoch : {epoch}, loss : {average_loss}") 
    #wandb.log({"loss": average_loss})
    # Step the learning rate scheduler after each epoch
    scheduler.step(average_loss)

grid_post = net.get_grid().cpu().sum(dim=1)
grid_res = np.where(grid_pre == grid_post, 1, 0)

# Save the model configuration after training is complete
print("Saving configuration")
current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'net_{current_date}_loss_{average_loss:.4f}.pth'
torch.save(net.state_dict(), 'saved/' + filename)