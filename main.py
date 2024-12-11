import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.net import Net
from src.training_loops import train_model
from src.data_preparation import prepare_data, prepare_tensors
from src.evaluation import evaluate_model, evaluate_robust_accuracy
from src.pgd_attack import pgd_attack, evaluate_impact_of_epsilon
from src.plots import plot_adversarial_examples, plot_confusion_matrix, plot_robust_accuracies
from src.fullynonnectednetwork import FullyConnectedNetwork
from src.robust_training import train_model_robust

def main():
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Hyperparameters
    learning_rate = 0.001
    momentum = 0.9
    epochs = 10
    batch_size = 64

    pgd_attack_params_evaluation = {
    "eps": 32 / 255,
    "alpha": (32 / 255) / 10,
    "n_iter": 50,
    "random_start": True
    }

    pgd_attack_params_training = {
        "eps": 32 / 255,
        "alpha": (32 / 255) / 5,
        "n_iter": 10,
        "random_start": True
    }

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 1.0: Prepare data
    print("Preparing data...")
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_data()
    x_train_tensor, x_val_tensor, x_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor = prepare_tensors(
        x_train, x_val, x_test, y_train, y_val, y_test
    )

    # Step 1.1: Initialize and train model_0 (CNN)
    print("\nInitializing and training Model 0 (Convolutional Neural Network)...")
    model_0 = Net()
    model_0.to(device)
    optimizer_0 = torch.optim.SGD(model_0.parameters(), lr=learning_rate, momentum=momentum)
    loss_func = torch.nn.CrossEntropyLoss()
    train_model(
        model=model_0,
        x_train=x_train_tensor,
        y_train=y_train_tensor,
        x_val=x_val_tensor,
        y_val=y_val_tensor,
        optimizer=optimizer_0,
        batch_size=batch_size,
        loss_func=loss_func,
        epochs=epochs
    )

    # Step 1.2: Evaluate clean accuracy
    print("\nEvaluating clean accuracy on the test set...")
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=2000, shuffle=False)
    clean_accuracy, conf_matrix_clean = evaluate_model(model_0, test_loader, device)
    print(f"Clean accuracy of Model 0: {clean_accuracy:.2f}%")
    class_names = [str(i) for i in range(10)]  # MNIST classes (0-9)
    plot_confusion_matrix(conf_matrix_clean, class_names, save_path="output/confusion_matrix_clean.png")

    # Step 1.3: Implement and execute the PGD attack on 1000 examples of the testing set
    # Step 1.4: Evaluate robust accuracy under PGD attack
    print("\nEvaluating robust accuracy under PGD attack...")
    attack_params = pgd_attack_params_evaluation
    test_subset = TensorDataset(x_test_tensor[:1000], y_test_tensor[:1000])
    test_loader_subset = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    robust_accuracy, conf_matrix_adv = evaluate_robust_accuracy(
        model=model_0,
        dataloader=test_loader_subset,
        attack_fn=pgd_attack,
        attack_params=attack_params,
        device=device
    )
    print(f"Robust accuracy under PGD attack: {robust_accuracy:.2f}%")
    plot_confusion_matrix(conf_matrix_adv, class_names, save_path="output/confusion_matrix_pgd.png")

    # Step 1.5: Plot adversarial examples
    print("\nGenerating and plotting adversarial examples...")
    visual_loader = DataLoader(test_subset, batch_size=10, shuffle=False)
    images, labels = next(iter(visual_loader))
    adv_images = pgd_attack(model_0, images.to(device), labels.to(device), **attack_params)
    plot_adversarial_examples(model_0, visual_loader, adv_images, class_names, device)

    # Step 1.6: Evaluate the impact of epsilon on robust accuracy for Model 0
    print("\nEvaluating the impact of epsilon on robust accuracy for Model 0...")
    eps_values, robust_accuracies_model_0 = evaluate_impact_of_epsilon(model_0, test_loader_subset, device)
    plot_robust_accuracies(
        eps_values,
        {"Model 0": robust_accuracies_model_0},
        "Impact of Epsilon on Robust Accuracy (Model 0)",
        "./output/impact_of_epsilon_model_0.png"
    )

    # Step 2.1 and 2.2: Initialize and train model_1 (Fully Connected Network)
    model_1 = FullyConnectedNetwork()
    model_1.to(device)
    optimizer_1 = torch.optim.SGD(model_1.parameters(), lr=learning_rate, momentum=momentum)
    train_model(
        model=model_1,
        x_train=x_train_tensor,
        y_train=y_train_tensor,
        x_val=x_val_tensor,
        y_val=y_val_tensor,
        optimizer=optimizer_1,
        batch_size=batch_size,
        loss_func=loss_func,
        epochs=epochs
    )

    # Step 2.3: Evaluate transferability of adversarial examples from Model 0 to Model 1
    print("\nEvaluating transferability of adversarial examples from Model 0 to Model 1...")
    model_1.eval()
    successful_on_model_0 = 0
    transferable_to_model_1 = 0
    total_adversarial_examples = len(adv_images)
    with torch.no_grad():
        for i in range(total_adversarial_examples):
            # Clean image and corresponding adversarial image
            clean_image = images[i].unsqueeze(0).to(device)
            adv_image = adv_images[i].unsqueeze(0).to(device)

            # Predictions on clean and adversarial images for model_0
            clean_pred_model_0 = model_0(clean_image).argmax(dim=1).item()
            adv_pred_model_0 = model_0(adv_image).argmax(dim=1).item()

            # Check if it fooled model_0
            if adv_pred_model_0 != clean_pred_model_0:
                successful_on_model_0 += 1

                # Predictions on adversarial image for model_1
                clean_pred_model_1 = model_1(clean_image).argmax(dim=1).item()
                adv_pred_model_1 = model_1(adv_image).argmax(dim=1).item()

                # Check if it also fools model_1
                if adv_pred_model_1 != clean_pred_model_1:
                    transferable_to_model_1 += 1

    # Calculate success and transferability ratios
    if successful_on_model_0 > 0:
        transferability_ratio = (transferable_to_model_1 / successful_on_model_0) * 100
        print(f"Transferability Ratio: {transferability_ratio:.2f}%")
    else:
        print("No successful adversarial examples on Model 0.")


    # Step 3.1: Adversarial Training of model_robust
    print("\nStep 3.1: Training Model Robust with Adversarial Training...")
    model_robust = Net()
    model_robust.to(device)
    optimizer_robust = torch.optim.SGD(model_robust.parameters(), lr=learning_rate, momentum=momentum)

    # PGD Attack Parameters for training
    attack_params = pgd_attack_params_training.copy()

    # Train model_robust with adversarial training
    history_robust = train_model_robust(
        model=model_robust,
        x_train=x_train_tensor,
        y_train=y_train_tensor,
        x_val=x_val_tensor,
        y_val=y_val_tensor,
        optimizer=optimizer_robust,
        batch_size=batch_size,
        loss_func=loss_func,
        epochs=2 * epochs,
        attack_fn=pgd_attack,
        attack_params=attack_params
    )
    print("\nEvaluating clean accuracy of Model Robust...")
    clean_accuracy_robust, _ = evaluate_model(model_robust, test_loader, device)
    print(f"Clean accuracy of Model Robust: {clean_accuracy_robust:.2f}%")
    print("\nEvaluating robust accuracy of Model Robust under PGD attack...")
    robust_accuracy_robust, conf_matrix_robust = evaluate_robust_accuracy(
        model=model_robust,
        dataloader=test_loader_subset,
        attack_fn=pgd_attack,
        attack_params=pgd_attack_params_evaluation,
        device=device
    )
    print(f"Robust accuracy of Model Robust: {robust_accuracy_robust:.2f}%")
    plot_confusion_matrix(conf_matrix_robust, class_names, save_path="output/confusion_matrix_robust.png")

    # Step 3.2:  
    print("\nEvaluating the impact of epsilon on robust accuracy for Model Robust...")
    eps_values, robust_accuracies_model_robust = evaluate_impact_of_epsilon(model_robust, test_loader_subset, device)

    plot_robust_accuracies(
        eps_values,
        {"Model 0": robust_accuracies_model_0, "Model Robust": robust_accuracies_model_robust},
        "Impact of Epsilon on Robust Accuracy (Comparison)",
        "./output/impact_of_epsilon_comparison.png"
    )

if __name__ == "__main__":
    main()
