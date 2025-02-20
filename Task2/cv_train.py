import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import sklearn.metrics as metrics
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definition image transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loading training and validation datasets
train_data = datasets.ImageFolder("data_cv/train", transform=transform)
valid_data = datasets.ImageFolder("data_cv/valid", transform=transform)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False)

# Loading pre-trained AlexNet model and modify the classifier for 18 output classes
net = models.alexnet(pretrained=True)
net.classifier[6] = nn.Linear(4096, 18)
net = net.to(device)

# Definition loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)


def train_model(model, train_loader, valid_loader, num_epochs=10):
    '''
    Model training using the animal dataset.

    Parameters:
    model: Neural network model
    train_loader: DataLoader for training data
    valid_loader: DataLoader for validation data
    num_epochs: Number of epochs to train
    '''
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)

        # Validation loop
        model.eval()
        valid_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        valid_loss /= len(valid_loader.dataset)
        valid_acc = 100. * correct / total

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Valid Acc: {valid_acc:.2f}%")


def evaluate_model_with_confusion_matrix(model, test_loader, save_path="confusion_matrix.png"):
    '''
    Evaluation the model on a test dataset and generate a confusion matrix.

    Parameters:
    model: Trained model
    test_loader: DataLoader for test data
    save_path (str): Path to save the confusion matrix image
    '''
    model.eval()
    all_labels = []
    all_preds = []

    # Collecting predictions and true labels
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Computing confusion matrix
    cm = metrics.confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_loader.dataset.classes,
                yticklabels=test_loader.dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Saving confusion matrix as an image
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    test_acc = 100. * np.sum(np.array(all_labels) == np.array(all_preds)) / len(all_labels)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Confusion matrix saved as {save_path}")


# Loading test dataset
test_data = datasets.ImageFolder("data_cv/test", transform=transform)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Model training (8 epochs) and saving
train_model(net, train_loader, valid_loader, num_epochs=8)
torch.save(net.state_dict(), "cv_model.pth")

# Model evaluation and generating confusion matrix
evaluate_model_with_confusion_matrix(net, test_loader, save_path="confusion_matrix.png")
