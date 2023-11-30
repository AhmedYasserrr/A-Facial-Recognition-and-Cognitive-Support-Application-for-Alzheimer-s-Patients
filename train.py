# Import standard dependencies
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf
from Preprocesing_and_Model_Engineering import *

# Create TensorFlow datasets for anchor, positive, and negative images
anchor = tf.data.Dataset.list_files("C:/Users/HP/OneDrive/Desktop/Facial Recognition/datam/anchor/*.jpg").take(400)
positive = tf.data.Dataset.list_files("C:/Users/HP/OneDrive/Desktop/Facial Recognition/datam/positive/*.jpg").take(400)
negative = tf.data.Dataset.list_files("C:/Users/HP/OneDrive/Desktop/Facial Recognition/datam/negative/*.jpg").take(400)

# Create Siamese Neural Network model, loss function, and optimizer
siamese_model = make_siamese_model()
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)  # Learning rate: 0.0001

# Set up checkpoint for model saving
checkpoint_dir = 'C:/Users/HP/OneDrive/Desktop/Facial Recognition/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

# Create positive and negative datasets for training
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

# Preprocess data using the preprocess_twin function
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10000)

# Split data into training and testing partitions
train_data = data.take(round(len(data) * 0.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

test_data = data.skip(round(len(data) * 0.7))
test_data = test_data.take(round(len(data) * 0.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

# Train the Siamese Neural Network model
EPOCHS = 20
train(siamese_model, binary_cross_loss, opt, checkpoint, checkpoint_prefix, train_data, EPOCHS)