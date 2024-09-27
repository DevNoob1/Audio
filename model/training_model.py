import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.preprocess import create_dataset_from_csv

# Load your labeled data from CSV
csv_file_path = "data/labeled_data.csv"
data = pd.read_csv(csv_file_path)

# Create train/test datasets using stratified split
train_df, test_df = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)

# Prepare datasets
train_dataset = create_dataset_from_csv(train_df).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = create_dataset_from_csv(test_df).batch(32).prefetch(tf.data.AUTOTUNE)

# Define the model architecture
input_shape = (None, 257, 1)   # Spectrogram shape
num_classes = 4  # Adjust based on your dataset (0, 1, 2, 3)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.GlobalAveragePooling2D(),  # Handles variable input lengths
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, epochs=4, validation_data=test_dataset)

# Save the model to the 'saved_model' directory
model.save("saved_model/audio_noise_classification_model.h5")
print("Model saved successfully!")
