# Image classification + OOP

This repository provide classifying on MNIST dataset using different algorithms. The MNIST dataset consists of 28x28 pixels images of handwritten digits (0-9).

## Features

- Supports three types of algorithms:
  - **Random Forest (rf)**
  - **Feed-Forward Neural Network (nn)**
  - **Convolutional Neural Network (cnn)**
- Command-line interface for training and evaluating models.

### Code Structure

- **`MnistClassifierInterface`**: Abstract base class defining the interface for all classifiers.
- **`RandomForestMnist`**: Implements a Random Forest classifier.
- **`FeedForwardMnist`**: Implements a Feed-Forward Neural Network.
- **`CNNMnist`**: Implements a Convolutional Neural Network.
- **`MnistClassifier`**: Wrapper class to easily use all models.

### Dataset Preparation

The MNIST dataset is automatically downloaded and preprocessed:
- Images are normalized to the range [0, 1].
- For Random Forest and FFNN, images are flattened into 1D vectors.
- For CNN, images are reshaped to include a channel dimension (28x28x1).

### Example Results

| Model       | Hyperparameters  | Accuracy  |
|-------------|-----------|-----------|
| Random Forest (rf) | n_estimators=100 (specifies the number of trees in the Random Forest)    | 96.90%    |
| Feed-Forward Neural Network (nn) | epochs=10, batch_size=64 | 97.79% |
| Convolutional Neural Network (cnn) | epochs=10, batch_size=64 | 98.52% |


## How to run

1. Clone the repository:

   ```bash
   git clone https://github.com/Silence-o0/Test-tasks
   cd Test-tasks/Task2
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train a model:
   
   To train a model, use the following command:

   ```bash
   python classifier.py [algo]
   ```

  Replace <algo> with one of the following options:
  - rf: Random Forest
  - nn: Feed-Forward Neural Network
  - cnn: Convolutional Neural Network

