import gc
import math
import torch
import wandb
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
from rich.progress import Progress
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

from network import NAF
from utils.params import load_params
from loaders.sound_loader import MeshrirDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'
params = load_params('params.yaml')

learning_rate = params['training']['lr']
epochs = params['training']['epochs']
batch_size = params['training']['batch']
ft = params['training']['ft']


print("Loading dataset ...")
dataset = MeshrirDataset(ft, Path('data/mesh_rir/S32-M441_npy/'), 0.0)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)


# Split dataloader in train, validation and test dataloaders
total_size = len(dataset)
val_size = int(0.1 * total_size)
test_size = int(0.1 * total_size)
train_size = total_size - val_size - test_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

# Spawn network and move to GPU
net = NAF(input_dim = 288, min_xy=dataset.min_pos[:2], max_xy=dataset.max_pos[:2]).to(device)

# Loss criterion
criterion = torch.nn.functional.l1_loss

# Initalize weight pools and create optimizer
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

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# Trackers
average_loss=0
print("Trainig started ...")

# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    net.train()

    with Progress() as progress:
        task = progress.add_task(f"[grey]Epoch {epoch+1}", total=len(train_loader))
        for batch_index, batch in enumerate(train_loader):
            # Unpack arrays and move to GPU
            gts_s = batch[0].to(device,  dtype=torch.float32)
            gts_p = batch[1].to(device, dtype=torch.float32)
            srcs = batch[2].to(device, dtype=torch.float32)
            mics = batch[3].to(device, dtype=torch.float32)
            freqs = batch[4].to(device, dtype=torch.float32)
            times = batch[5].to(device, dtype=torch.float32)

            output = net(srcs, mics, freqs, times)
            s_loss = criterion(gts_s, output[:,:,0])
            p_loss = criterion(gts_p, output[:,:,1])
            a = 0.5
            loss = (a*s_loss)+(1-a)*p_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress.update(task, advance=1)

    # Validation
    net.eval()
    val_error = 0

    with torch.no_grad():
        for batch in val_loader:
            gts_s = batch[0].to(device,  dtype=torch.float32)
            gts_p = batch[1].to(device, dtype=torch.float32)
            srcs = batch[2].to(device, dtype=torch.float32)
            mics = batch[3].to(device, dtype=torch.float32)
            freqs = batch[4].to(device, dtype=torch.float32)
            times = batch[5].to(device, dtype=torch.float32)

            output = net(srcs, mics, freqs, times)
            s_loss = criterion(gts_s, output[:,:,0])
            p_loss = criterion(gts_p, output[:,:,1])
            a = 0.5
            val_error += (a*s_loss)+(1-a)*p_loss

    average_loss = (running_loss / len(train_loader))/batch_size
    val_error = (val_error / len(val_loader))/batch_size
    scheduler.step(average_loss)

    print(f"  → Training error: {average_loss}")
    print(f"  → Validation error: {val_error}")

del train_dataset, train_loader
del val_dataset, val_loader
torch.cuda.empty_cache()
gc.collect()


# # Test loop
# net.eval()
# test_error = 0
# for batch in test_loader:
#     gts_s = batch[0].to(device,  dtype=torch.float32)
#     gts_p = batch[1].to(device, dtype=torch.float32)
#     srcs = batch[2].to(device, dtype=torch.float32)
#     mics = batch[3].to(device, dtype=torch.float32)
#     freqs = batch[4].to(device, dtype=torch.float32)
#     times = batch[5].to(device, dtype=torch.float32)
#     output = net(srcs, mics, freqs, times)
#     s_loss = criterion(gts_s, output[:,:,0])
#     p_loss = criterion(gts_p, output[:,:,1])
#     a = 0.5
#     test_error += (a*s_loss)+(1-a)*p_loss

# test_error = (test_error / len(test_loader))/batch_size
# print(f"  → Test error: {test_error}")

# Save the model configuration after training is complete
current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'model_{current_date}_{average_loss:.7f}.pth'
print(f"Saving configuration as {filename}")
torch.save(net.state_dict(), 'saved/' + filename)
