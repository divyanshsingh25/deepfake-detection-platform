import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC


IMG_SIZE = 260
BATCH_SIZE = 16
EPOCHS = 12

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# ---------------- DATA AUGMENTATION ----------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# ---------------- BASE MODEL ----------------
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

# ---------------- CUSTOM HEAD ----------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", AUC(name="auc")]
)

model.summary()

# -------- CLASS WEIGHTS (FIX IMBALANCE) --------
# real = 0, fake = 1
class_weights = {0: 1.0, 1: 1.0}

# Auto-adjust if imbalanced
total_real = train_generator.classes.tolist().count(0)
total_fake = train_generator.classes.tolist().count(1)

if total_real != total_fake:
    class_weights = {
        0: total_fake / (total_real + total_fake),
        1: total_real / (total_real + total_fake)
    }

print("Using class weights:", class_weights)


# ---------------- TRAIN ----------------
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights
)


MODEL_PATH = os.path.join(BASE_DIR, "model", "deepfake_model.h5")
model.save(MODEL_PATH)
print("✅ Improved model saved at", MODEL_PATH)

print("✅ Improved model saved as deepfake_model.h5")
