# Import standard dependencies
import os
import numpy as np
import csv
from matplotlib import pyplot as plt

# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf


def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img

def preprocess_twin(input_img_path, validation_img_path, label):
    """
    Preprocess twin images and their label.

    Parameters:
    - input_img_path: File path of the input image
    - validation_img_path: File path of the validation image
    - label: Label for the images

    Returns:
    - Preprocessed input image tensor, preprocessed validation image tensor, and label
    """
    return(preprocess(input_img_path), preprocess(validation_img_path), label)

def make_embedding():
    """
    Create a Siamese Neural Network embedding model.

    Returns:
    - Siamese Neural Network embedding model
    """ 
    inp = Input(shape=(100,100,3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')

# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def make_siamese_model(embedding = make_embedding()): 
    """
    Create a Siamese Neural Network model.

    Parameters:
    - embedding: Siamese Neural Network embedding model

    Returns:
    - Siamese Neural Network model
    """
    input_image = Input(name='input_img', shape=(100,100,3))
    validation_image = Input(name='validation_img', shape=(100,100,3))

    # Combine siamese distance components
    siamese_Dist_layer = L1Dist()
    distances = siamese_Dist_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


@tf.function
def train_step(batch, siamese_model, binary_cross_loss, opt):
    # Record all of our operations 
    with tf.GradientTape() as tape:     
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        
        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)
        
    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
        
    # Return loss
    return loss


def append_to_csv(file_path, data):
    # Check if the CSV file already exists
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Read the header
    except FileNotFoundError:
        # If the file doesn't exist, create it with a header
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            header = ['Loss', 'Recall', 'Precision']
            writer.writerow(header)

    # Append the data to the CSV file
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def train(siamese_model, binary_cross_loss, opt, checkpoint, checkpoint_prefix, data, EPOCHS):
    # Loop through epochs
    best = 1000000 # just a high num
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        # Creating a metric object 
        r = Recall()
        p = Precision()
        
        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            loss = train_step(batch, siamese_model, binary_cross_loss, opt)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat) 
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())
        
        # Save checkpoints
        if epoch % 1 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            # Create a list with the data to be appended
            data_to_append = [loss.numpy(), r.result().numpy(), p.result().numpy()]
            # Append the data to the CSV file
            append_to_csv("C:/Users/HP/OneDrive/Desktop/Facial Recognition/results.csv", data_to_append)
            if loss.numpy() < best:
                best = loss.numpy()
                # Save weights
                siamese_model.save('C:/Users/HP/OneDrive/Desktop/Facial Recognition/best.h5') 
                

def verify(model, detection_threshold = 0.5, verification_threshold= 0.5):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('C:/Users/HP/OneDrive/Desktop/Facial Recognition/application_data', 'verification_images')):
        input_img = preprocess(os.path.join('C:/Users/HP/OneDrive/Desktop/Facial Recognition/application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('C:/Users/HP/OneDrive/Desktop/Facial Recognition/application_data', 'verification_images', image))
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(os.path.join('C:/Users/HP/OneDrive/Desktop/Facial Recognition/application_data', 'verification_images'))) 
    # verified = verification > verification_threshold
    
    return results, verification