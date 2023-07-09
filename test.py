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

from modules import embedding_module_log

test = torch.from_numpy(np.linspace(0.0, 0.4, 2).astype(np.single)).unsqueeze(0)
print(test)
print(test.shape)
emb = embedding_module_log(test).expand(-1, 4, -1)
res = emb(test)

print(res)
print(res.shape)