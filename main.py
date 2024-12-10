import os
from net import Net
from training_loops import train_model
from data_preparation import prepare_data, prepare_tensors
import torch
from torch.utils.data import TensorDataset, DataLoader
from evaluation import evaluate_model, evaluate_robust_accuracy
from pgd_attack import pgd_attack, evaluate_impact_of_epsilon
from plots import plot_adversarial_examples

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

    accuracy, _ = evaluate_model(model_0, test_loader, device)
    print(f"Clean accuracy of the model is {accuracy:.2f}%")

    # PGD Attack Parameters
    attack_params = {
        "eps": 32 / 255,
        "alpha": (32 / 255) / 10,
        "n_iter": 50,
        "random_start": True
    }

    print("Evaluating robust accuracy under PGD attack...")
    test_subset = TensorDataset(x_test_tensor[:1000], y_test_tensor[:1000])
    test_loader_subset = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    robust_accuracy, conf_matrix = evaluate_robust_accuracy(
        model_0, test_loader_subset, pgd_attack, attack_params, device
    )
    print(f"Robust accuracy under PGD attack: {robust_accuracy:.2f}%")

    # Evaluate and plot adversarial examples
    visual_loader = DataLoader(test_subset, batch_size=10, shuffle=False)
    images, labels = next(iter(visual_loader))
    adv_images = pgd_attack(model_0, images.to(device), labels.to(device), **attack_params)

    class_names = [str(i) for i in range(10)]  # MNIST classes (0-9)
    plot_adversarial_examples(model_0, visual_loader, adv_images, class_names, device)

    # Evaluate impact of epsilon
    print("\nEvaluating the impact of epsilon on robust accuracy...")
    evaluate_impact_of_epsilon(model_0, test_loader_subset, device)


if __name__ == "__main__":
    main()
