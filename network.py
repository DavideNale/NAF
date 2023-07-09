import torch
import torch.nn as nn
import numpy as np

class NAF(nn.Module):
    def __init__(self, input_dim, hidden_dim = 512, output_dim = 1, grid_density = 0.25,
        feature_dim = 64, min_xy = None, max_xy = None):
        super(NAF, self).__init__()

        # Create grid
        grid_coords_x = np.arange(min_xy[0], max_xy[0], grid_density)
        grid_coords_y = np.arange(min_xy[1], max_xy[1], grid_density)
        

        
        # Define layers        
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, 512)
        self.skip1 = nn.Linear(512, 512)
        self.skip2 = nn.Linear(512, 512)
        self.relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        x1 = self.relu(self.fc1(x))
        x2 = self.relu(self.fc2(x1))
        x3 = self.relu(self.fc3(x2))
        x4 = self.relu(self.fc4(x3))
        x_skip = self.relu(self.skip1(x))
        x4_skip = x4 + x_skip
        x5 = self.relu(self.fc5(x4_skip))
        x6 = self.relu(self.fc6(x5))
        x7 = self.relu(self.fc7(x6))
        x8 = self.relu(self.fc8(x7))
        x_skip2 = self.relu(self.skip2(x4_skip))
        output = x8 + x_skip2
        return output