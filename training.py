import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle

# Define parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20

# Path to your dataset (ensure it has 'Normal' and 'Monkeypox' subfolders)
DATASET_DIR = 'Dataset'  # Update this path to the correct dataset location

# Load images and labels
def load_images_and_labels(dataset_dir):
    images = []
    labels = []
    class_names = os.listdir(dataset_dir)  # List subfolder names (e.g., 'Normal' and 'Monkeypox')
    class_names.sort()  # Sort to ensure correct labeling (Normal -> 0, Monkeypox -> 1)
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            try:
                # Load image, resize, and normalize
                img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                img_array = img_to_array(img) / 255.0  # Normalize pixel values
                images.append(img_array)
                labels.append(label)  # Use 0 for 'Normal', 1 for 'Monkeypox'
            except Exception as e:
                print(f"Error loading image {img_file}: {e}")
    
    images, labels = shuffle(np.array(images), np.array(labels), random_state=42)
    return images, labels

# Load the dataset
X, y = load_images_and_labels(DATASET_DIR)
print(f"Loaded {len(X)} images from {DATASET_DIR}. Classes: {np.unique(y)}")

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Save the trained model
model.save('cnn_monkeypox_model.h5')

print("Model training completed and saved as 'cnn_monkeypox_model.h5'.")
