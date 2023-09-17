import numpy as np
import torch
import torch.optim as optim
import math
from torch.utils.data import DataLoader
import datetime
import wandb

from network import NAF
from sound_loader import sound_samples
from modules import positional_encoding, WeightedMSELoss

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Trainig parameters
learning_rate = 0.0005
num_epochs = 50
batch_size = 20
ft_num = 200

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
net = NAF(input_dim = 416, min_xy=dataset.min_pos[:2], max_xy=dataset.max_pos[:2]).to(device)
criterion = torch.nn.MSELoss()#WeightedMSELoss()#torch.nn.MSELoss()

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

# print('Calculating weight scaling bins ...')
# # Spectrograms
# hist, bins_s = np.histogram((dataset.spectrograms + 40) / 40, bins=10, density=True)
# #hist = np.clip(hist, 0, 0.4)
# hist_normalized = hist / float(np.sum(hist))
# mean = np.mean(hist)
# spectros_probs = np.ones_like(hist) - mean + hist
# # Phases
# hist, bins_p = np.histogram(dataset.phases / 180, bins=20, density=True)
# hist_normalized = hist / float(np.sum(hist))
# mean = np.mean(hist)
# phases_probs = np.ones_like(hist) - mean + hist

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0

    for batch_index, batch in enumerate(dataloader):
        # Unpack arrays and move to GPU
        gts_s = (batch[0].to(device, dtype=torch.float32) + 40) / 40
        gts_p = batch[1].to(device, dtype=torch.float32)
        #gts_p = torch.sin(0.5 * math.pi * gts_p)
        #gts_p = torch.tanh(gts_p * 3)
        gts = torch.cat((gts_s.unsqueeze(2), gts_p.unsqueeze(2)), dim=2)
        srcs = batch[2].to(device, dtype=torch.float32)
        mics = batch[3].to(device, dtype=torch.float32)
        freqs = batch[4].to(device, dtype=torch.float32)
        times = batch[5].to(device, dtype=torch.float32)

        # Feed to the network
        output = net(srcs, mics, freqs, times)
        #output[:,:,1] = torch.tanh(output[:,:,1] * 3)
        # output_s = - output[:,:,0].detach().unsqueeze(2).cpu().numpy()
        # output_p = output[:,:,1].detach().unsqueeze(2).cpu().numpy()

        # digitized_indices = np.digitize(gts_s.detach().clone().cpu().flatten(), bins_s) - 2
        # digitized_indices[digitized_indices >= len(bins_s)] = len(bins_s) - 1
        # weights_s = spectros_probs[digitized_indices]
        # weights_s = weights_s.reshape(output_s.shape)
        # weights_s = torch.tensor(weights_s, requires_grad=False)
        # # weights_s = torch.ones(output_s.shape, requires_grad=False)

        # digitized_indices = np.digitize(gts_p.detach().clone().cpu().flatten(), bins_p) - 2
        # digitized_indices[digitized_indices >= len(bins_p)] = len(bins_p) - 1
        # weights_p = phases_probs[digitized_indices]
        # weights_p = weights_p.reshape(output_p.shape)
        # weights_p = torch.tensor(weights_p, requires_grad=False)
        # # weights_p = torch.ones(output_p.shape, requires_grad=False)

        # weights = torch.cat((weights_s, weights_p), dim=-1).to(device)
        # # output = output * weights

        # Calculate loss
        # loss = criterion(gts, -output, weights)
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
filename = f'net_{current_date}_loss_{average_loss:.7f}.pth'
torch.save(net.state_dict(), 'saved/' + filename)