import os
import pandas as pd

# Define paths
noise_folder = 'C:/Users/nikha/OneDrive/Desktop/dev/ML/dataset/data/Noise'
non_noise_folder = 'C:/Users/nikha/OneDrive/Desktop/dev/ML/dataset/data/Non_Noise'

# Define labels
labels = {
    'Car_Horns': 1,
    'Drilling': 2,
    'Jackhammar': 3,
    'traffic': 0,
    'ESC': 0
}

# Initialize a list to hold the file paths and labels
data = []

# Traverse noise folder
for label_name, label in labels.items():
    folder_path = os.path.join(noise_folder if label != 0 else non_noise_folder, label_name)
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):  # or .mp3, depending on your file format
                data.append({'file_path': os.path.join(folder_path, file_name), 'label': label})

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('labeled_data.csv', index=False)
