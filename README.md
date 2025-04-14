
# Medical MNIST Image Classification

This project focuses on classifying medical images from the Medical MNIST dataset available on Kaggle. The goal was to train a Convolutional Neural Network (CNN) using PyTorch to accurately identify different body parts from MRI and CT scans.

## Project Overview

This repository contains the code and documentation for training and evaluating a CNN model on a dataset of 58,954 medical images. The dataset includes MRI and CT scans of the following body parts:

* Abdomen
* Hand
* Breast
* CXR (Chest X-ray)
* Head

The project involved the following key steps:

1.  **Dataset Loading:** Loading the Medical MNIST dataset from Kaggle.
2.  **Image Processing and Scaling:** Preprocessing the images to ensure they are suitable for training. This likely involved normalization or other scaling techniques.
3.  **Data Loading and Batching:** Utilizing PyTorch's `Dataset` and `DataLoader` classes to efficiently load and batch the data with a batch size of 32. This facilitates efficient training of the neural network.
4.  **CNN Model Training:** Building and training a CNN model using PyTorch. The model architecture is defined within the code.
5.  **Hyperparameter Tuning:** Defining training parameters such as the number of epochs, loss function, optimizer, and learning rate.
6.  **Model Evaluation:** Evaluating the trained model on both the training and testing datasets. The model achieved an impressive **93.3% accuracy** on test set.
7.  **Prediction and Visualization:** Implementing a prediction system to test the trained model on new, unseen images and visualizing the prediction results.

## Dataset

The Medical MNIST dataset, containing 58,954 images, was sourced from Kaggle. It comprises MRI and CT scans categorized into five classes: Abdomen, Hand, Breast, CXR, and Head.

## Technologies Used

* Python
* PyTorch
* Keras and Tensorflow
* PIL
* Numpy, Pandas, Matplotlib, Seaborn and Scikit Learn. 
