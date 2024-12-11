import torch
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on a given dataset and print performance metrics.

    Parameters:
    - model (torch.nn.Module): Model to evaluate.
    - dataloader (DataLoader): DataLoader for the evaluation dataset.
    - device (torch.device): Device to run the evaluation on.

    Returns:
    - float: Accuracy of the model on the dataset (percentage).
    - np.ndarray: Confusion matrix.
    """
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculations
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            pred_classes = preds.argmax(dim=1)

            # Accumulate correct predictions and total samples
            correct += (pred_classes == y).sum().item()
            total += y.size(0)

            # Store predictions and labels for confusion matrix
            all_preds.extend(pred_classes.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Calculate accuracy
    accuracy = (correct / total) * 100

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(f"\nClassification Report:\n{classification_report(all_labels, all_preds)}")

    return accuracy, conf_matrix


def evaluate_robust_accuracy(model, dataloader, attack_fn, attack_params, device):
    """
    Evaluate the model's robust accuracy under a specified adversarial attack.

    Parameters:
    - model (torch.nn.Module): The trained model to evaluate.
    - dataloader (DataLoader): DataLoader for the dataset to attack.
    - attack_fn (callable): Function to generate adversarial examples (e.g., pgd_attack).
    - attack_params (dict): Parameters for the attack function.
    - device (torch.device): Device to perform computations on.

    Returns:
    - float: Robust accuracy (%).
    - list: All predicted labels for adversarial examples.
    - list: All true labels for the corresponding adversarial examples.
    """
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    model.eval()  # Set the model to evaluation mode

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Enable gradients temporarily for attack
        with torch.enable_grad():
            # Generate adversarial examples using the attack function
            adv_images = attack_fn(model, images, labels, **attack_params)

        # Disable gradients for evaluation
        with torch.no_grad():
            # Evaluate the model on the adversarial examples
            outputs = model(adv_images)
            pred_classes = outputs.argmax(dim=1)

            # Count correct predictions
            correct += (pred_classes == labels).sum().item()
            total += labels.size(0)

            # Store predictions and true labels
            all_preds.extend(pred_classes.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate robust accuracy as a percentage
    robust_accuracy = (correct / total) * 100
    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return robust_accuracy, conf_matrix



