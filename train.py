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



# Instantiate network

# Dataset
print('Loading dataset. It might take a while ...')
dataset = sound_samples()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
print(dataset.min_xy)
print(dataset.max_xy)
xyz_embedder = embedding_module_log().to(device)
freq_embedder = embedding_module_log().to(device)
time_embedder = embedding_module_log().to(device)

net = NAF().to(device)

print('Training started: ')

# for each batch ( 20 spects)
# random sample 2000 (f,t)
# join with position in a single tensor
# trainig and optimize over it

for batch_index, batch in enumerate(dataloader):
    print("batch: ",batch_index)
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
    

exit(0)

# an epoch is a comlpete visit of the dataset
for epoch in range(num_epochs):
    for batch in dataloader:

        # do things with the data
        with torch.no_grad():
            pos_embed = ''
            freq_embed = ''
            time_embed = ''

        total_in = torch.cat((pos_embed, freq_embed, time_embed), dim=2)

        # and continue the optimization
    

"""
for batch, samples in enumerate(dataloader):
    print(f'=== BATCH {i_batch} ============')
    src, mic, spectrogram = sample
    # process ir
    # infer from net
    # compare
    # optimize net

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
