import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import make_siamese_model, triplet_loss
from data_loader import load_dataset, prepare_data
from config import LEARNING_RATE, CHECKPOINT_PREFIX, EPOCHS

def train_model():
    """
    Trains the Siamese model using Triplet Loss.
    """
    # Load and prepare dataset
    anchor, positive, negative = load_dataset()
    train_data, test_data = prepare_data(anchor, positive, negative)

    # Create model
    siamese_model = make_siamese_model()
    optimizer = Adam(LEARNING_RATE)

    # Compile with triplet loss
    siamese_model.compile(optimizer=optimizer, loss=triplet_loss())

    # Train model
    siamese_model.fit(train_data, epochs=EPOCHS, validation_data=test_data)

    # Save checkpoint
    siamese_model.save(CHECKPOINT_PREFIX)
    print(f"Model saved at {CHECKPOINT_PREFIX}")

if __name__ == "__main__":
    train_model()
