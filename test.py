import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import soundfile as sf
import random
import torch
import librosa

import utils.irutilities as irutil

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

from numpy.lib.format import header_data_from_array_1_0

filename = 'mesh_rir/S32-M441_npy/spectrograms.npy'

file = np.load(filename)

print(file[0,0,0,0].dtype)