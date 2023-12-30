import numpy as np
from dataclasses import dataclass

@dataclass
class InputData:
    magnitude_gt: np.ndarray
    phase_gt: np.ndarray
    srcs_norm: np.ndarray
    mic_norm: np.ndarray
    freqs_norm: np.ndarray
    times_norm: np.ndarray
