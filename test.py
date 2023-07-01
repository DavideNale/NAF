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

from sound_loader import sound_samples

dataset = sound_samples()
src, mic, ir = dataset[0]
print(src)
print(mic)
print(ir)

_, _, IRs = irutil.loadIR(Path('mesh_rir/S32-M441_npy/'))
print(len(IRs))
