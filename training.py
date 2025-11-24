import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ---------- PATHS ----------
DATASET_PATH = r"C:\Users\hp\PycharmProjects\PythonProject21\dataset"
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 15


# ---------- DATA GENERATORS ----------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,

    # Anti-spoofing augmentations
    zoom_range=0.2,
    brightness_range=[0.6, 1.3],
    rotation_range=10,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)


# ---------- BUILD MODEL ----------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # Freeze backbone

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ---------- TRAIN MODEL ----------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)


# ---------- SAVE MODEL ----------
model.save("anti_spoof_mobilenet.h5")
print("✔ Model saved as anti_spoof_mobilenet.h5")
