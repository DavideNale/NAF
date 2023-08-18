import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt

from sound_loader import sound_samples
from utilities import plot_images

dataset = sound_samples(num_samples=20000)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4, persistent_workers=True)

batch = next(iter(dataloader))
gts = batch[0]
freqs = batch[3]
times = batch[4]

images = []
for i in range(20):
    images.append(np.zeros((1025,65)))

for b in range(freqs.shape[0]):
    for s in range(freqs.shape[1]):
        images[b][int(1025*(1/2*freqs[b,s]+0.5)), int(65*(1/2*times[b,s]+0.5))] = gts[b,s]

plot_images(images, 2, 5, (100,100))

