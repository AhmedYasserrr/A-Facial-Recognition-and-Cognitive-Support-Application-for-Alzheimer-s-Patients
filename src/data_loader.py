import tensorflow as tf
import os
import numpy as np
from config import ANCHOR_DIR, POSITIVE_DIR, NEGATIVE_DIR, BATCH_SIZE, BUFFER_SIZE

def load_dataset():
    """
    Loads dataset from the given directories.
    """
    anchor = tf.data.Dataset.list_files(os.path.join(ANCHOR_DIR, "*.jpg")).take(400)
    positive = tf.data.Dataset.list_files(os.path.join(POSITIVE_DIR, "*.jpg")).take(400)
    negative = tf.data.Dataset.list_files(os.path.join(NEGATIVE_DIR, "*.jpg")).take(400)

    return anchor, positive, negative

def preprocess_twin(anchor_path, other_path, label):
    """
    Loads and preprocesses images for training.
    """
    def load_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.resize(image, (105, 105))
        image = image / 255.0  # Normalize
        return image

    return load_image(anchor_path), load_image(other_path), label

def prepare_data(anchor, positive, negative):
    """
    Prepares the dataset by applying preprocessing and batching.
    """
    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    
    data = positives.concatenate(negatives)
    data = data.map(preprocess_twin)
    data = data.shuffle(BUFFER_SIZE)
    
    # Split into train and test datasets
    train_size = int(0.7 * len(data))
    train_data = data.take(train_size).batch(BATCH_SIZE).prefetch(8)
    test_data = data.skip(train_size).batch(BATCH_SIZE).prefetch(8)
    
    return train_data, test_data
