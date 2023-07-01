import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from network import NAF
from sound_loader import sound_samples
torch.manual_seed(42)

# Network parameters
feature_dim = 64
hidden_dim = 256
output_dim = 100

# Trainig parameters
learning_rate = 0.001
num_epochs = 200
batch_size = 20

# Feature grid definition



# Instantiate network

# Dataset
print('Loading dataset. It might take a while ...')
dataset = sound_samples()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print('Training started: ')

# for each batch ( 20 spects)
# random sample 2000 (f,t)
# join with position in a single tensor
# trainig and optimize over it


"""
for i_batch, sample in enumerate(dataloader):
    print(f'=== BATCH {i_batch} ============')
    print('yas')
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
