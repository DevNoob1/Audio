import tensorflow as tf
import pandas as pd
import numpy as np
from utils.load_wav import load_wav_16k_mono 

TARGET_SPECTROGRAM_LENGTH = 257 

def preprocess(file_path, label):
    file_path = file_path.numpy().decode("utf-8")
    wav = load_wav_16k_mono(file_path)
    

    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)

    current_length = tf.shape(spectrogram)[0]
    if current_length < TARGET_SPECTROGRAM_LENGTH:
        padding = TARGET_SPECTROGRAM_LENGTH - current_length
        spectrogram = tf.pad(spectrogram, [[0, padding], [0, 0]], mode='CONSTANT')
    elif current_length > TARGET_SPECTROGRAM_LENGTH:
        spectrogram = spectrogram[:TARGET_SPECTROGRAM_LENGTH, :]

    spectrogram = tf.expand_dims(spectrogram, axis=-1) 
    label = tf.cast(label, tf.int32)
    return spectrogram, label

def create_dataset_from_csv(dataframe):
    dataset = tf.data.Dataset.from_tensor_slices((dataframe['file_path'].values, dataframe['label'].values))
    
    def map_fn(file_path, label):
        spectrogram, lbl = tf.py_function(func=preprocess, inp=[file_path, label], Tout=[tf.float32, tf.int32])
        spectrogram.set_shape([TARGET_SPECTROGRAM_LENGTH, None, 1]) 
        lbl.set_shape([])  
        return spectrogram, lbl
    
    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset
