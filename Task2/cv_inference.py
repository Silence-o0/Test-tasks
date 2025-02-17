from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cv_inference(image_path):
    """
    Image classification using a pre-trained AlexNet model.

    Args:
        image_path (str): Path to the input image.

    Returns:
        str or None: Predicted class label if confidence is above threshold, otherwise None.
    """
    threshold = 0.5

    # Loading class names from the training dataset
    dataset = datasets.ImageFolder("data_cv/train")
    class_names = dataset.classes

    # Initializing AlexNet model with a modified classifier layer
    model = models.alexnet(weights=None)
    model.classifier[6] = torch.nn.Linear(4096, 18)

    model_path = "cv_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at ./{model_path}")
        return None

    # Loading the trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Loading and preprocessing the input image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Inferencing
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)

    # Check confidence level
    if confidence.item() < threshold:
        print("Unknown animal. The model can't recognize it.")
        return None
    return class_names[predicted.item()]

