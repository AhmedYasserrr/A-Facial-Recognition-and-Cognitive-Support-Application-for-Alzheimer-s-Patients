import os

# Dataset paths
DATASET_DIR = "data"
ANCHOR_DIR = os.path.join(DATASET_DIR, "anchor")
POSITIVE_DIR = os.path.join(DATASET_DIR, "positive")
NEGATIVE_DIR = os.path.join(DATASET_DIR, "negative")

# Training parameters
BATCH_SIZE = 16
BUFFER_SIZE = 10000
EPOCHS = 20
LEARNING_RATE = 1e-4

# Checkpoint settings
CHECKPOINT_DIR = "models/checkpoints"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")
