import tensorflow as tf
import numpy as np
from utils.load_wav import load_wav_16k_mono  # Import load_wav function

# Load the saved model
model = tf.keras.models.load_model("saved_model/audio_noise_classification_model.h5")

# Preprocess the audio segment for inference
def preprocess_segment(wav_segment):
    spectrogram = tf.signal.stft(wav_segment, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    
    # Pad the spectrogram to a consistent length
    target_length = 257  # Set this according to your model's input size
    current_length = tf.shape(spectrogram)[0]
    
    if current_length < target_length:
        padding = target_length - current_length
        spectrogram = tf.pad(spectrogram, [[0, padding], [0, 0]], mode='CONSTANT')
    elif current_length > target_length:
        spectrogram = spectrogram[:target_length, :]
        
    spectrogram = tf.expand_dims(spectrogram, axis=-1)  # Add channel dimension
    spectrogram = tf.expand_dims(spectrogram, axis=0)  # Add batch dimension
    return spectrogram

# Inference function to detect noise and collect timestamps
def detect_noise_in_audio(file_path, segment_duration=1.0, overlap_duration=0.5):
    wav = load_wav_16k_mono(file_path)
    
    # Parameters for segmenting the audio
    sample_rate = 16000
    segment_length = int(segment_duration * sample_rate)
    overlap_length = int(overlap_duration * sample_rate)
    
    noise_segments = []

    for start in range(0, len(wav) - segment_length + 1, segment_length - overlap_length):
        segment = wav[start:start + segment_length]
        spectrogram = preprocess_segment(segment)
        
        predictions = model.predict(spectrogram)
        predicted_label = np.argmax(predictions)
        
        if predicted_label != 0:  # Non-noise
            start_time = start / sample_rate
            end_time = (start + segment_length) / sample_rate
            
            noise_segments.append((start_time, end_time))
    
    # Summarize noise periods
    results = []
    if noise_segments:
        # Initialize with the first noise segment
        current_start, current_end = noise_segments[0]
        
        for start_time, end_time in noise_segments[1:]:
            if start_time <= current_end:  # Overlapping or contiguous
                current_end = max(current_end, end_time)
            else:  # No overlap
                results.append((current_start, current_end, 'Noise Detected'))
                current_start, current_end = start_time, end_time
        
        # Append the last segment
        results.append((current_start, current_end, 'Noise Detected'))

    return results

# Test with a new audio file
file_path = "data/test/test1.wav"  # Updated file path
results = detect_noise_in_audio(file_path)

# Format output
for start_time, end_time, status in results:
    print(f"{status}: from {start_time:.2f} seconds to {end_time:.2f} seconds")
