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

def plot_confusion_matrix(conf_matrix, class_names):
    """
    Plot the confusion matrix.

    Parameters:
    - conf_matrix (np.ndarray): Confusion matrix.
    - class_names (list): List of class names for the dataset.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    # Add tick marks and labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add counts to each cell in the matrix
    thresh = conf_matrix.max() / 2.0
    for i, j in np.ndindex(conf_matrix.shape):
        plt.text(
            j, i, format(conf_matrix[i, j], "d"),
            horizontalalignment="center",
            color="white" if conf_matrix[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("./output/confusion_matrix.png")

def evaluate_robust_accuracy(model, dataloader, attack_fn, attack_params, device):
    """
    Evaluate the model's robust accuracy under a specified adversarial attack.

    Parameters:
    - model (torch.nn.Module): The trained model to evaluate.
    - dataloader (DataLoader): DataLoader for the dataset to attack.
    - attack_fn (callable): Function to generate adversarial examples.
    - attack_params (dict): Parameters for the attack function.
    - device (torch.device): Device to perform computations on.

    Returns:
    - float: Robust accuracy (%).
    """
    correct = 0
    total = 0

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for images, labels in dataloader:
            # Move data to the appropriate device
            images, labels = images.to(device), labels.to(device)

            # Generate adversarial examples
            adv_images = attack_fn(model, images, labels, **attack_params)

            # Evaluate the model on adversarial examples
            outputs = model(adv_images)
            pred_classes = outputs.argmax(dim=1)

            # Count correct predictions
            correct += (pred_classes == labels).sum().item()
            total += labels.size(0)

    # Calculate robust accuracy
    robust_accuracy = (correct / total) * 100
    return robust_accuracy

