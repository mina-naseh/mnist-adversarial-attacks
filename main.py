import os
from net import Net
from training_loops import train_model
from data_preparation import prepare_data, prepare_tensors
import torch
from torch.utils.data import TensorDataset, DataLoader
from evaluation import evaluate_model, plot_confusion_matrix, evaluate_robust_accuracy
from pgd_attack import pgd_attack

def main():
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Hyperparameters
    learning_rate = 0.001
    momentum = 0.9
    epochs = 2
    batch_size = 64

    # Prepare data
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_data()
    x_train_tensor, x_val_tensor, x_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor = prepare_tensors(
        x_train, x_val, x_test, y_train, y_val, y_test
    )

    # Initialize model
    model_0 = Net()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_0.to(device)

    # Initialize optimizer and loss function
    optimizer = torch.optim.SGD(model_0.parameters(), lr=learning_rate, momentum=momentum)
    loss_func = torch.nn.CrossEntropyLoss()

    # Train the model
    train_model(
        model=model_0,
        x_train=x_train_tensor,
        y_train=y_train_tensor,
        x_val=x_val_tensor,
        y_val=y_val_tensor,
        optimizer=optimizer,
        batch_size=batch_size,
        loss_func=loss_func,
        epochs=epochs
    )

    # Evaluate clean accuracy
    print("\nEvaluating model on the test set...")
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=2000, shuffle=False)

    accuracy, conf_matrix = evaluate_model(model_0, test_loader, device)
    print(f"Clean accuracy of the model is {accuracy:.2f}%")

    # Plot confusion matrix
    class_names = [str(i) for i in range(10)]  # MNIST classes (0-9)
    plot_confusion_matrix(conf_matrix, class_names)


    # PGD Attack Parameters
    eps = 32 / 255
    alpha = eps / 10
    n_iter = 50
    n_examples = 1000

    # Select a subset of the test set
    test_subset = TensorDataset(x_test_tensor[:n_examples], y_test_tensor[:n_examples])
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    print("Generating adversarial examples using PGD...")

    # Evaluate model performance on adversarial examples
    model_0.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Generate adversarial examples
        adv_images = pgd_attack(model_0, images, labels, eps=eps, alpha=alpha, n_iter=n_iter)

        # Evaluate the model on adversarial examples
        outputs = model_0(adv_images)
        pred_classes = outputs.argmax(dim=1)

        # Count correct predictions
        correct += (pred_classes == labels).sum().item()
        total += labels.size(0)

    # Calculate robust accuracy
    robust_accuracy = (correct / total) * 100
    print(f"Robust accuracy under PGD attack: {robust_accuracy:.2f}%")


if __name__ == "__main__":
    main()
