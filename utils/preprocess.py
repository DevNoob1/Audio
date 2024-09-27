import tensorflow as tf
import pandas as pd
import numpy as np
from utils.load_wav import load_wav_16k_mono  # Import load wav function

# Set a target length for the spectrogram
TARGET_SPECTROGRAM_LENGTH = 257  # Adjust this based on your analysis of typical spectrogram sizes

def preprocess(file_path, label):
    file_path = file_path.numpy().decode("utf-8")
    wav = load_wav_16k_mono(file_path)
    
    # Compute the spectrogram for the full-length audio
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)

    # Pad or truncate the spectrogram to the target length
    current_length = tf.shape(spectrogram)[0]
    if current_length < TARGET_SPECTROGRAM_LENGTH:
        # Pad the spectrogram if it's too short
        padding = TARGET_SPECTROGRAM_LENGTH - current_length
        spectrogram = tf.pad(spectrogram, [[0, padding], [0, 0]], mode='CONSTANT')
    elif current_length > TARGET_SPECTROGRAM_LENGTH:
        # Truncate the spectrogram if it's too long
        spectrogram = spectrogram[:TARGET_SPECTROGRAM_LENGTH, :]

    spectrogram = tf.expand_dims(spectrogram, axis=-1)  # Add channel dimension
    label = tf.cast(label, tf.int32)
    return spectrogram, label

# Create TensorFlow dataset from CSV
def create_dataset_from_csv(dataframe):
    dataset = tf.data.Dataset.from_tensor_slices((dataframe['file_path'].values, dataframe['label'].values))
    
    def map_fn(file_path, label):
        spectrogram, lbl = tf.py_function(func=preprocess, inp=[file_path, label], Tout=[tf.float32, tf.int32])
        spectrogram.set_shape([TARGET_SPECTROGRAM_LENGTH, None, 1])  # Set expected shape for spectrogram
        lbl.set_shape([])  # Set expected shape for label (scalar)
        return spectrogram, lbl
    
    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset
