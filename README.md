# A-Facial-Recognition-and-Cognitive-Support-Application-for-Alzheimer-s-Patients
A Comprehensive Facial Recognition and Cognitive Support Application for Alzheimer's Patients, Leveraging Siamese Neural Networks for One-shot Image Recognition, Providing Location and Direction Guidance, Along with an Visual To-Do List all implemented on Google Glasses platform.



# Siamese Neural Network for Facial Recognition

## Overview
This project implements a **Siamese Neural Network** for facial recognition using **Triplet Loss**. The model learns to differentiate between similar and dissimilar faces, making it ideal for verification tasks.

## Features
- **Siamese Network Architecture**: Uses convolutional layers to extract features from images.
- **Triplet Loss Optimization**: Ensures the model minimizes the distance between similar faces while maximizing it for different faces.
- **Efficient Data Handling**: Loads and processes anchor, positive, and negative image pairs.
- **Model Training & Evaluation**: Trains with a structured pipeline and evaluates performance.

## Installation
### Prerequisites
Ensure you have **Python 3.8+** and the following dependencies installed:
```bash
pip install tensorflow numpy matplotlib
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/siamese-face-recognition.git
cd siamese-face-recognition
```

## Project Structure
```
├── model.py                # Siamese Network model definition
├── data_loader.py          # Data loading and preprocessing functions
├── train.py                # Model training script
├── config.py               # Configurations for training
├── README.md               # Project documentation
```

## Training the Model
To train the model, run:
```bash
python train.py
```
The trained model will be saved at the checkpoint location specified in `config.py`.

## Configuration
Modify **config.py** to adjust hyperparameters such as:
```python
LEARNING_RATE = 0.001
EPOCHS = 10
CHECKPOINT_PREFIX = "saved_model/best_model.h5"
```

## Model Verification
You can test the trained model using the `verify()` function in `model.py`, which compares an input image with stored validation images.

## Future Enhancements
- **Improve dataset augmentation** to enhance robustness.
- **Optimize for edge devices** like Raspberry Pi.
- **Deploy as an API** using Flask or FastAPI.

## Contributing
Feel free to open issues and submit pull requests!

## License
This project is licensed under the **MIT License**.

