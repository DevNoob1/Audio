import os
from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
from flask_cors import CORS
from utils.load_wav import load_wav_16k_mono  # Your utility function

UPLOAD_FOLDER = 'uploads'  # Folder to store uploaded files

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

# Load the saved model
model = tf.keras.models.load_model("saved_model/audio_noise_classification_model.h5")

def preprocess_segment(wav_segment):
    spectrogram = tf.signal.stft(wav_segment, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    
    target_length = 257  # Set this according to your model's input size
    current_length = tf.shape(spectrogram)[0]
    
    if current_length < target_length:
        padding = target_length - current_length
        spectrogram = tf.pad(spectrogram, [[0, padding], [0, 0]], mode='CONSTANT')
    elif current_length > target_length:
        spectrogram = spectrogram[:target_length, :]
        
    spectrogram = tf.expand_dims(spectrogram, axis=-1)  
    spectrogram = tf.expand_dims(spectrogram, axis=0)  
    return spectrogram

def detect_noise_in_audio(wav, segment_duration=1.0, overlap_duration=0.5):
    sample_rate = 16000
    segment_length = int(segment_duration * sample_rate)
    overlap_length = int(overlap_duration * sample_rate)
    
    noise_segments = []

    for start in range(0, len(wav) - segment_length + 1, segment_length - overlap_length):
        segment = wav[start:start + segment_length]
        spectrogram = preprocess_segment(segment)
        
        if spectrogram.shape[1] == 0:
            continue
        
        predictions = model.predict(spectrogram)
        predicted_label = np.argmax(predictions)
        
        if predicted_label != 0:  # Assuming label 0 is "no noise"
            start_time = start / sample_rate
            end_time = (start + segment_length) / sample_rate
            noise_segments.append((start_time, end_time))
    
    results = []
    if noise_segments:
        current_start, current_end = noise_segments[0]
        for start_time, end_time in noise_segments[1:]:
            if start_time <= current_end:  
                current_end = max(current_end, end_time)
            else:
                results.append((current_start, current_end, 'Noise Detected'))
                current_start, current_end = start_time, end_time
        results.append((current_start, current_end, 'Noise Detected'))

    return results

def results_to_json(results):
    output = []
    for start_time, end_time, status in results:
        output.append({
            "status": status,
            "start_time": f"{start_time:.2f}",
            "end_time": f"{end_time:.2f}"
        })
    return output

@app.route('/detect-noise', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.wav'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)  # Save the file
        
        wav = load_wav_16k_mono(file_path)  # Load the audio data from the saved file
        
        if wav is None or len(wav) == 0:
            return jsonify({"error": "Failed to load audio data."}), 400
        
        results = detect_noise_in_audio(wav)  # Process the file
        json_results = results_to_json(results)

        return jsonify({
            "file_url": f"http://localhost:5000/uploads/{file.filename}",
            "noise_segments": json_results
        })  # Return the file URL and JSON results
    
    return jsonify({"error": "Invalid file format. Only .wav files are accepted."}), 400

# Serve uploaded files
@app.route('/uploads/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(port=5000, debug=True)
