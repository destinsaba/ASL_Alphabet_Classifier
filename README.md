# ASL Alphabet Classifier

This project implements a neural network that recognizes the American Sign Language (ASL) alphabet. The model is trained on a dataset of images representing the ASL alphabet and can classify images into one of the 29 classes (26 letters of the alphabet, 'del', 'nothing', and 'space').

## Dataset

The model is trained on the ASL Alphabet dataset, which can be found on Kaggle: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data).

The dataset has been split into two main folders:
- `asl_alphabet_train`: Contains training images organized into subfolders for each class.
- `asl_alphabet_test`: Contains test images organized into subfolders for each class.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- PIL
- scikit-learn
- matplotlib
- tqdm
- wandb

## Setup

1. Clone the repository

2. Install the required packages

3. Set up Weights & Biases (wandb) for experiment tracking

## Training the Model
To train the model, run the `classifier.py` script. This script will train the model on the ASL Alphabet dataset and save the best model based on validation loss. Alternatively, download the previously-trained best_model here: [https://drive.google.com/file/d/1L0SPc-Wms54PWnAQkrNqIPBqRL94efO9/view?usp=sharing]

## Evaluating the Model
To evaluate the model, use the model_evaluation.ipynb Jupyter Notebook. This notebook will load the best model, evaluate it on the test dataset, and provide various metrics such as accuracy, confusion matrix, and classification report.

## Live Testing with Webcam
To test the classifier with your computer's webcam, run the live_test.py script. This script will capture video from the webcam, classify each frame, and display the predicted class on the screen.

## Results
The model achieves high accuracy on the test dataset. Detailed results, including accuracy, confusion matrix, and classification report, can be found in the model_evaluation.ipynb notebook.

## Acknowledgements
- The ASL Alphabet dataset is provided by Kaggle.
- This project uses PyTorch and torchvision for building and training the neural network.
- Experiment tracking is done using Weights & Biases.
