import glob
import json
import torch
import numpy.random
import os
import pickle
import numpy as np
import random
import h5py
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)



class soundsamples(torch.utils.data.Dataset):

    def __init__(self, folder_name):

        self.data = []
        self.sources = []
        self.mics = []

        folder_path = os.path.join("dataset", folder_name)
        mp3_files = glob.glob(os.path.join(folder_path, "*.wav"))

        coords = None
        with open('dataset/data.json') as file:
            coords = json.load(file)

        # Extract the source positions
        src = []
        for key, value in coords.items():
            if key.startswith("src_"):
                src.append(value["pos"])

        # Extract the microphone positions
        mic = []
        for key, value in coords.items():
            if key.startswith("ir_"):
                mic.append(value["pos"])

        for file_path in mp3_files:

            # extract src and mic coordinates
            file_name, _ = os.path.splitext(os.path.basename(file_path))
            name_parts = file_name.split("_")
            src_idx = int(name_parts[-2])
            mic_idx = int(name_parts[-1])

            # loading sound
            element = (file_name, src[src_idx], mic[mic_idx])

            # here we have to extract src and mic
            self.data.append(element)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample[0], sample[1], sample[2]  # and then positions
