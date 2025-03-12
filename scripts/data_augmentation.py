# Import standard dependencies
import cv2
import os
import random
import numpy as np
import uuid
import tensorflow as tf

def data_aug(img):
    """
    Apply data augmentation techniques to an input image.

    Parameters:
    - img: Input image tensor

    Returns:
    - List of augmented image tensors
    """
    data = []
    for i in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,3))
      
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9,upper=1, seed=(np.random.randint(100),np.random.randint(100)))
            
        data.append(img)
    
    return data

# Define paths for positive, negative, and anchor images
POS_PATH = os.path.join('C:/Users/HP/OneDrive/Desktop/Facial Recognition/datam', 'positive')
NEG_PATH = os.path.join('C:/Users/HP/OneDrive/Desktop/Facial Recognition/datam', 'negative')
ANC_PATH = os.path.join('C:/Users/HP/OneDrive/Desktop/Facial Recognition/datam', 'anchor')

# Apply data augmentation to anchor images
for file_name in os.listdir(os.path.join(ANC_PATH)):
    img_path = os.path.join(ANC_PATH, file_name)
    img = cv2.imread(img_path)
    augmented_images = data_aug(img) 
    
    for image in augmented_images:
        # Save augmented images with unique filenames using UUID
        cv2.imwrite(os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())


# Apply data augmentation to positive images
for file_name in os.listdir(os.path.join(POS_PATH)):
    img_path = os.path.join(POS_PATH, file_name)
    img = cv2.imread(img_path)
    augmented_images = data_aug(img) 
    
    for image in augmented_images:
        # Save augmented images with unique filenames using UUID
        cv2.imwrite(os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())
