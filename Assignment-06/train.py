import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================
# Dataset Paths
# ==========================

train_dir = "tomato/train"
val_dir = "tomato/val"

# ==========================
# Data Preprocessing
# ==========================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical"
)

# ==========================
# CNN Model
# ==========================

model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(128, 128, 3)
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(
    Conv2D(
        128,
        (3, 3),
        activation="relu"
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dropout(0.5))

model.add(
    Dense(
        train_data.num_classes,
        activation="softmax"
    )
)

# ==========================
# Compile Model
# ==========================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# Model Summary
# ==========================

model.summary()

# ==========================
# Train Model
# ==========================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# ==========================
# Evaluate Model
# ==========================

loss, accuracy = model.evaluate(val_data)

print("\nValidation Loss :", loss)
print("Validation Accuracy :", accuracy)

# ==========================
# Save Model
# ==========================

model.save("tomato_disease_cnn.keras")

print("\nModel saved successfully.")

# ==========================
# Plot Accuracy Graph
# ==========================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()