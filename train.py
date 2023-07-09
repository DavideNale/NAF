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

# Feature grid definition




# Dataset
print('Loading dataset. It might take a while ...')
dataset = sound_samples()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
print('min: ', dataset.min_xy)
print('max: ', dataset.max_xy)

# Spawn embedders and move to GPU
xyz_embedder = embedding_module_log().to(device)
freq_embedder = embedding_module_log().to(device)
time_embedder = embedding_module_log().to(device)

# Spawn network and move to GPU
net = NAF(input_dim = 126, min_xy=dataset.min_xy, max_xy=dataset.max_xy).to(device)
criterion = torch.nn.MSELoss()

# Create pools for optimization
orig_container = []
grid_container = []

for par_name, par_val in net.named_parameters():
    print(par_name)
    #if "grid" in par_name:
        #grid_container.append(par_val)
    #else:
        #orig_container.append(par_val)

optimizer = torch.optim.AdamW([
    {'params': grid_container, 'lr': learning_rate, 'weight_decay': 1e-2},
    {'params': grid_container, 'lr': learning_rate, 'weight_decay': 0}],
    lr = learning_rate,
    weight_decay = 0.0
)

print('Training started: ')

# for each batch ( 20 spects)
# random sample 2000 (f,t)
# join with position in a single tensor
# trainig and optimize over it

# Training loop
for epoch in range(num_epochs):
    for batch_index, batch in enumerate(dataloader):
        print('batch: ', batch_index)
        srcs, mics, spectrograms = batch


        for i in range(srcs.shape[0]):
            # exctract single data
            src = srcs[i,:]
            mic = mics[i,:]
            
            # pick ft_num (f,t) from each spectrogram 
            for i in range(ft_num):
                f_max, t_max = spectrogram.shape
                f = torch.randint(0,f_max, size=(1,))
                t = torch.randint(0, t_max, size=(1,))
                
                    
        # for each row of the 20
        # extract positions
        # extract spectrogram
        # selecgt 2000 (f,t)
        # compute FF
        # compect in single array
        # compact in tensor
    
        # for each line of the tensor train the network
        # do things with the data

        with torch.no_grad():
            pos_embed = ''
            freq_embed = ''
            time_embed = ''

        input = torch.cat((pos_embed, freq_embed, time_embed), dim=2)
        optimizer.zero_grad(set_to_none=Flase)

        output = net(input,)

        loss = criterion(output, ground_truth)
        loss.backward()
        optimizer.step()



"""

for epoch in range(num_epochs):
    for coordinates, features, targets in dataloader:
        # Forward pass
        output = network(coordinates, features)
        
        # Calculate loss
        loss = loss_fn(output, targets)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")


# Create example feature and coordinate tensors
feature_tensor = torch.randn(batch_size, feature_dim)
coordinate_tensor = torch.randn(batch_size, 3)
target_response_tensor = torch.randn(batch_size, output_dim)

# Instantiate the network
network = SpatialAudioNetwork(feature_dim, hidden_dim, output_dim)

# Define loss function and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for coordinates, features, targets in dataloader:
        # Forward pass
        output = network(coordinates, features)
        
        # Calculate loss
        loss = loss_fn(output, targets)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
"""
