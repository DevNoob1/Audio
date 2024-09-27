import librosa
import numpy as np

def load_wav_16k_mono(filename):
    """
    Load an audio file as a mono waveform with a sample rate of 16kHz.
    
    Args:
        filename (str): Path to the audio file.
    
    Returns:
        np.ndarray: Loaded audio waveform.
    """
    try:
        # Load the audio file with a sample rate of 16kHz
        wav, _ = librosa.load(filename, sr=16000, mono=True)
        return wav

    except Exception as e:
        print(f"Error loading {filename}: {e}")
        # Return silence of target_length if file loading fails
        return np.zeros(48000)  # You can adjust this default length if necessary
