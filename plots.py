import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_confusion_matrix(conf_matrix, class_names, save_path="./output/confusion_matrix.png"):
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
    plt.savefig(save_path)


def plot_adversarial_examples(
    model, dataloader, adv_images, class_names, device, n_classes=10
):
    """
    Plot original and adversarial examples for one image per class.

    Parameters:
    - model (torch.nn.Module): Trained model to predict labels.
    - dataloader (DataLoader): DataLoader containing the original images and labels.
    - adv_images (torch.Tensor): Generated adversarial examples.
    - class_names (list): Class names (0–9 for MNIST).
    - device (torch.device): Device for computations.
    - n_classes (int): Number of classes to represent in the plot.
    """
    original_images = {}
    adversarial_images = {}
    original_preds = {}
    adversarial_preds = {}

    model.eval()

    # Loop through the dataloader to find one image per class
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Predict original and adversarial classes
        with torch.no_grad():
            orig_outputs = model(images)
            adv_outputs = model(adv_images)
            orig_classes = orig_outputs.argmax(dim=1)
            adv_classes = adv_outputs.argmax(dim=1)

        # Store one image per class
        for i in range(len(labels)):
            label = labels[i].item()
            if label not in original_images:
                original_images[label] = images[i].detach().cpu()  # Detach and move to CPU
                adversarial_images[label] = adv_images[i].detach().cpu()  # Detach and move to CPU
                original_preds[label] = orig_classes[i].item()
                adversarial_preds[label] = adv_classes[i].item()
            # Stop if we have one image for each class
            if len(original_images) == n_classes:
                break
        if len(original_images) == n_classes:
            break

    # Plot the images
    plt.figure(figsize=(15, 10))
    for i, label in enumerate(class_names):
        # Original image
        plt.subplot(2, n_classes, i + 1)
        plt.imshow(original_images[i].squeeze(), cmap="gray")
        plt.title(f"Orig: {label}\nPred: {original_preds[i]}")
        plt.axis("off")

        # Adversarial image
        plt.subplot(2, n_classes, i + 1 + n_classes)
        plt.imshow(adversarial_images[i].squeeze(), cmap="gray")
        plt.title(f"Adv Pred: {adversarial_preds[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("./output/adversarial_examples_per_class.png")
    print("Plot saved to ./output/adversarial_examples_per_class.png")
