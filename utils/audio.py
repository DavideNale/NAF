import librosa
import numpy as np


# Extracts amplitude (dB) and phase (unwrapped) from an audio sample
def from_audio(sample: np.ndarray, n_fft: int, hop: int) -> (np.ndarray, np.ndarray):
    complex = librosa.stft(sample, n_fft=n_fft, hop_length=hop)
    spectrogram = np.abs(complex)
    spectrogram_db = librosa.amplitude_to_db(spectrogram)
    phase = np.angle(complex)
    phase_unwrapped = np.unwrap(phase)
    return spectrogram_db, phase_unwrapped

# converts amplitude + phase into audio
def to_audio(spectrogram_db, phase_unwrapped) -> np.ndarray:
    return librosa.istft(
        librosa.db_to_amplitude(spectrogram_db) * np.exp(1j * phase_unwrapped),
        hop_length=512 // 4,
    )
