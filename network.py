import torch
import torch.nn as nn

class NAF(nn.Module):
    def __init__(self, grid_size, feature_dim, hidden_dim, output_dim):
        super(NAF, self).__init__()
        
        # Network layers
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(feature_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, coordinates, features):
        x = self.relu(self.fc1(coordinates))
        x = self.relu(self.fc2(x))
        y = self.relu(self.fc3(features))
        y = self.relu(self.fc4(y))
        output = self.fc5(y)
        return output
