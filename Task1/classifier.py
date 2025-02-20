import argparse
import numpy as np
from tensorflow.keras.datasets import mnist
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input


class MnistClassifierInterface(ABC):
    '''
    Abstract base class that defines the interface for MNIST classifiers.
    Any subclass must implement the `train` and `predict` methods.
    '''
    @abstractmethod
    def train(self, X, y):
        '''
        Trains the model on the provided data.

        Parameters:
            X (numpy.ndarray): The training data (features).
            y (numpy.ndarray): The target labels.
        '''
        pass

    @abstractmethod
    def predict(self, X):
        '''
        Predicts the class labels for the provided data.

        Parameters:
            X (numpy.ndarray): The input data (features).

        Returns:
            numpy.ndarray: The predicted class labels.
        '''
        pass


class RandomForestMnist(MnistClassifierInterface):
    '''
    This class use Random Forest algorithm for classification.
    '''
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class FeedForwardMnist(MnistClassifierInterface):
    '''
    This class use Feed-Forward Neural Network for classification.
    '''
    def __init__(self):
        self.model = Sequential([
            Input(shape=(784,)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X, y):
        self.model.fit(X, y, epochs=10, batch_size=64, verbose=2)

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)


class CNNMnist(MnistClassifierInterface):
    '''
    This class use simple Convolutional Neural Network  with one conv layer for classification.
    '''
    def __init__(self):
        self.model = Sequential([
            Input(shape=(28, 28, 1)),
            Conv2D(32, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X, y):
        self.model.fit(X, y, epochs=10, batch_size=64, verbose=2)

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)


class MnistClassifier:
    '''
    A wrapper class to easily switch between different classifiers (Random Forest,
    Feed-Forward Neural Network, Convolutional Neural Network).
    '''
    def __init__(self, algorithm):
        '''
        Parameters:
            algorithm (str): The algorithm to use. Options: 'rf' (Random Forest),
            'nn' (Feed-Forward Neural Network), 'cnn' (Convolutional Neural Network).
        '''
        if algorithm == 'rf':
            self.model = RandomForestMnist()
        elif algorithm == 'nn':
            self.model = FeedForwardMnist()
        elif algorithm == 'cnn':
            self.model = CNNMnist()
        else:
            raise ValueError("Unknown algorithm. Choose 'rf', 'nn', or 'cnn'.")

    def train(self, X, y):
        self.model.train(X, y)

    def predict(self, X):
        return self.model.predict(X)


if __name__=="__main__":
    # Parsing input arguments (algorithm) from terminal

    parser = argparse.ArgumentParser(description="Train an MNIST classifier.")
    parser.add_argument("algo", help="Algorithm to use: cnn, rf, or nn")
    args = parser.parse_args()

    algo = args.algo
    if algo not in ['cnn', 'rf', 'nn']:
        print("Error: Unknown algorithm. Please choose 'cnn', 'rf', or 'nn'.")

    else:
        print(f"Training {algo} model...")

        # Load MNIST dataset and normalize pixel values to the range [0, 1]
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train, X_test = X_train / 255.0, X_test / 255.0

        # Prepare data for models
        X_train_flat = X_train.reshape(-1, 28 * 28)  # Flatten the images for Random Forest and FFNN
        X_test_flat = X_test.reshape(-1, 28 * 28)
        X_train_cnn = X_train.reshape(-1, 28, 28, 1)  # Reshape for CNN (add channel dimension)
        X_test_cnn = X_test.reshape(-1, 28, 28, 1)

        # Train and evaluate models
        model = MnistClassifier(algo)
        if algo == 'cnn':
            model.train(X_train_cnn, y_train)
            predictions = model.predict(X_test_cnn)
        elif algo == 'nn':
            model.train(X_train_flat, y_train)
            predictions = model.predict(X_test_flat)
        else:
            model.train(X_train_flat, y_train)
            predictions = model.predict(X_test_flat)

        # Calculate accuracy
        accuracy = np.mean(predictions == y_test)
        print(f"{algo} accuracy: {accuracy * 100:.2f}%")
