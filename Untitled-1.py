# %%
# from tensorflow.keras.mixed_precision import set_global_policy
# set_global_policy('mixed_float16')


# %%
import os
import numpy as np
import pydicom

def load_dicom_series(root_folder):
    """
    Recursively loads all DICOM files from nested folders and organizes them into patient-wise volumes.
    """
    patient_volumes = {}  # Dictionary to store volumes per patient

    for dirpath, _, filenames in os.walk(root_folder):  # Walk through all subdirectories
        dicom_files = sorted([f for f in filenames if f.endswith(".dcm")])  # Sort for correct order

        if dicom_files:  # If the folder contains DICOM files
            patient_id = dirpath.split("/")[-2]  # Extract patient ID from folder name
            slices = [pydicom.dcmread(os.path.join(dirpath, f)).pixel_array for f in dicom_files]
            
            volume = np.stack(slices, axis=-1)  # Stack slices to form 3D volume
            
            patient_volumes[patient_id] = volume

    return patient_volumes

# Example usage
root_folder = "PPMI"  # Adjust to your dataset's root directory
spect_volumes = load_dicom_series(root_folder)

# Print details
for patient, volume in spect_volumes.items():
    print(f"Patient: {patient}, Shape: {volume.shape}")  # Example output: (128, 128, 64)


# %%
import cv2

def preprocess_volume(volume, target_size=(64, 64, 64)):
    """
    Normalizes and resizes the 3D volume.
    """
    # Normalize (Scale values between 0 and 1)
    volume = volume.astype(np.float32) / np.max(volume)

    # Resize each 2D slice correctly
    # Remove the extra channel dimension if present
    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = np.squeeze(volume, axis=-1)

    resized_slices = [cv2.resize(slice, target_size[:2]) for slice in volume[..., 0]]
    resized_volume = np.stack(resized_slices, axis=-1)  # Stack resized slices

    return np.expand_dims(resized_volume, axis=-1)  # Add channel dimension

# Example preprocessing

spect_volumes = dict(list(spect_volumes.items())[:100])
preprocessed_volumes = {pid: preprocess_volume(vol) for pid, vol in spect_volumes.items()}


# %%
import numpy as np
import cv2
import random

# Ensure all volumes have the same shape
volume_shape = (64, 64, 64, 1)  # Target shape

X_train = []
for vol in preprocessed_volumes.values():
    vol = np.squeeze(vol)  # Remove unnecessary singleton dimensions

    # Ensure the depth is correct
    num_slices = vol.shape[-1]
    if num_slices != 64:
        resized_slices = []
        for i in range(64):  # Always create exactly 64 slices
            idx = int(i * num_slices / 64)  # Sample indices proportionally
            resized_slices.append(cv2.resize(vol[:, :, idx], volume_shape[:2]))  
        vol = np.stack(resized_slices, axis=-1)  # Stack into depth

    vol = np.expand_dims(vol, axis=-1)  # Ensure (64, 64, 64, 1)
    X_train.append(vol)

X_train = np.array(X_train)  # Convert list to numpy array

# Generate random labels and one-hot encode them
y_train = np.array([random.randint(0, 1) for _ in range(len(X_train))])  # Random binary labels

print("X_train shape:", X_train.shape)  # Should be (N, 64, 64, 64, 1)
print("y_train shape:", y_train.shape)  # Should be (N,)


# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization

# Model architecture
model = Sequential([
    Conv3D(32, kernel_size=(3,3,3), activation='relu', input_shape=(64, 64, 64, 1)),
    BatchNormalization(),
    MaxPooling3D(pool_size=(2,2,2)),

    Conv3D(64, kernel_size=(3,3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling3D(pool_size=(2,2,2)),

    Conv3D(128, kernel_size=(3,3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling3D(pool_size=(2,2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train
history = model.fit(X_train, y_train, epochs=10, batch_size=2, validation_split=0.2)


# %%


# %%


# %%
import pandas as pd

# %%
data = pd.read_csv('/home/antriksh/Desktop/DAV PROJECT/mil_gaya_4_02_2025.csv')
data

# %%
data["Group"].value_counts()

# %%



