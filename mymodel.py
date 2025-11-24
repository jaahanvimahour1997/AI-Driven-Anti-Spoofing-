import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# PATHS
DATASET_PATH = r"C:\Users\hp\PycharmProjects\PythonProject21\dataset"  # change if needed
IMG_SIZE = 96
BATCH_SIZE = 32

# Data Augmentation for Anti-Spoofing
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,

    # Anti-spoofing specific augmentations
    zoom_range=0.2,
    brightness_range=[0.6, 1.3],
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

# Train Set
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Validation Set
val_generator = test_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print("\n✔ Dataset Loaded Successfully")
print("👉 Classes:", train_generator.class_indices)
