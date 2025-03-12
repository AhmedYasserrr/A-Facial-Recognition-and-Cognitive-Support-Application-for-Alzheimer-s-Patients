import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Input
from tensorflow.keras.models import Model

def make_siamese_model():
    """
    Creates a Siamese neural network model for learning embeddings.
    """
    input_shape = (105, 105, 3)
    input_layer = Input(shape=input_shape, name="input_image")
    
    x = Conv2D(64, (10,10), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(128, (7,7), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(128, (4,4), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(256, (4,4), activation='relu')(x)
    x = Flatten()(x)
    output_layer = Dense(128, activation='linear')(x)  # 128-dimensional embedding
    
    return Model(inputs=input_layer, outputs=output_layer, name="Siamese_Model")

def euclidean_distance(vectors):
    """
    Computes the Euclidean distance between two vectors.
    """
    anchor, other = vectors
    return tf.sqrt(tf.reduce_sum(tf.square(anchor - other), axis=1, keepdims=True))

def triplet_loss(alpha=1):
    """
    Triplet loss function with margin α.
    """
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        loss = tf.maximum(pos_dist - neg_dist + alpha, 0.0)
        return tf.reduce_mean(loss)
    
    return loss
