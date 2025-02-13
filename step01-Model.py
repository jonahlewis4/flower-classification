import tensorflow as tf
import keras

TRAINING_PATH = "./data/training"

train_dataset = keras.utils.image_dataset_from_directory(
    TRAINING_PATH,
    validation_split=0.2
)