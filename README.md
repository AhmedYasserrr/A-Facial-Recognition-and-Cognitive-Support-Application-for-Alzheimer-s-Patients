# Facial Recognition and Cognitive Support for Alzheimer's Patients using Siamese Neural Networks
## Overview
This project utilizes a Siamese Neural Network trained with Triplet Loss for facial recognition. The model is designed to distinguish between similar and dissimilar faces, making it well-suited for identity verification tasks. This technology can assist Alzheimer's patients in recognizing their relatives and close persons, helping them navigate social interactions more effectively.

## Features
- **Siamese Network Architecture**: Uses convolutional layers to extract features from images.
- **Triplet Loss Optimization**: Ensures the model minimizes the distance between similar faces while maximizing it for different faces.
- **Efficient Data Handling**: Loads and processes anchor, positive, and negative image pairs.

## Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/siamese-face-recognition.git
cd siamese-face-recognition
```
### Prerequisites
Ensure you have **Python 3.8+** and the following dependencies installed:
```bash
pip install -r requirements.txt
```
## Project Structure
```
├── src
│   ├── model.py          # Siamese Network model definition
│   ├── data_loader.py    # Data loading and preprocessing functions
│   ├── train.py          # Model training script
│   ├── config.py         # Configuration settings for training
│
├── scripts
│   ├── data_collection.py   # Collect face images using a webcam for local data gathering
│   ├── data_augmentation.py # Apply augmentation techniques to enhance training data
│
├── data
│   ├── anchor              
│   ├── positive           
│   ├── negative          
│
├── reports
│   ├── report.pdf        # Project report
│
├── README.md             # Project documentation
├── requirements.txt      # List of dependencies

```
### Data Organization  
Before training, ensure that your dataset is structured correctly. The model relies on three types of images:  

- **Anchor**: The reference image of a person.  
- **Positive**: Another image of the same person (similar to the anchor).  
- **Negative**: An image of a different person (dissimilar to the anchor).  

These images should be placed in the following directories:  
```
data/
│── anchor/    # Reference images
│── positive/  # Similar images
│── negative/  # Dissimilar images
```
By default, `train.py` loads data from these folders. However, you can change the dataset location by modifying `config.py`.

## Training the Model

To train the model, run:  
```bash
python train.py
```
The trained model will be saved at the checkpoint location specified in `config.py`.

## Configuration
Modify **config.py** to adjust hyperparameters such as:
```python
DATASET_DIR = "data"
LEARNING_RATE = 0.001
EPOCHS = 10
CHECKPOINT_PREFIX = "saved_model/best_model.h5"
```

