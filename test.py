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


from modules import embedding_module_log
from sound_loader import sound_samples

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sound = sound_samples(num_samples=20)

print(sound[1])
#test = torch.from_numpy(np.linspace(0.0, 0.4, 2).astype(np.single)).unsqueeze(0)
#print(test)
#
#emb = embedding_module_log(num_freqs=10)
#res = emb(test)
#
#print(res)
#print(res.shape)
