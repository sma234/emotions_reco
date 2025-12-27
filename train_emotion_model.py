import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 25

def build_emotion_model():
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(48,48,1)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(len(EMOTIONS), activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    base_dir = "dataset"  

    if not os.path.exists(base_dir):
        raise Exception("Folder 'dataset' not found next to train_emotion_model.py")

    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        base_dir,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        classes=EMOTIONS,         
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = build_emotion_model()
    model.summary()

   
    checkpoint = ModelCheckpoint(
        "emotion_model.h5",
        monitor="accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    model.fit(
        train_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop]
    )

    model.save("emotion_model_last.h5")
    print("Training finished and best model saved as emotion_model.h5")

if __name__ == "__main__":
    main()
