import librosa
import numpy as np

def preprocess_audio(file_path):
    """
    Load and preprocess an audio file.
    """
    audio, sr = librosa.load(file_path, sr=48000, mono=True) # Load audio file at a sample rate of 48kHz and convert to mono, which is a single channel
    audio = audio / np.max(np.abs(audio))  # Normalize amplitude to be between -1 and 1 which helps with training
    return audio, sr