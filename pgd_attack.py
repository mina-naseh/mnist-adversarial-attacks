import torch
import torch.nn.functional as F
from torch import nn, Tensor
from evaluation import evaluate_robust_accuracy

def pgd_attack(
    model: nn.Module, 
    images: Tensor, 
    labels: Tensor, 
    eps: float = 32/255, 
    alpha: float = (32/255)/10, 
    n_iter: int = 50, 
    random_start: bool = True
) -> Tensor:
    """
    Perform the Projected Gradient Descent (PGD) attack.

    Parameters:
    - model (torch.nn.Module): The model to attack.
    - images (torch.Tensor): Input images (normalized), shape [batch_size, channels, height, width].
    - labels (torch.Tensor): Ground truth labels for the input images, shape [batch_size].
    - eps (float): Maximum perturbation (epsilon) for each pixel.
    - alpha (float): Step size for each iteration.
    - n_iter (int): Number of attack iterations.
    - random_start (bool): Whether to initialize adversarial examples randomly within the epsilon ball.

    Returns:
    - torch.Tensor: Adversarial examples, same shape as input images.
    """
    # Validate input shapes
    if images.dim() != 4 or labels.dim() != 1:
        raise ValueError("Expected images to have 4 dimensions (B, C, H, W) and labels to have 1 dimension (B).")
    if images.size(0) != labels.size(0):
        raise ValueError("Batch size of images and labels must match.")

    device = images.device
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    # Initialize adversarial examples
    if random_start:
        adv_images = images + torch.empty_like(images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, 0, 1)
    else:
        adv_images = images.clone()

    adv_images.requires_grad = True

    for _ in range(n_iter):
        # Zero gradients from the previous iteration
        if adv_images.grad is not None:
            adv_images.grad.zero_()

        # Compute the loss
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        # Compute gradients
        loss.backward()

        # Update adversarial images with gradient ascent
        grad = adv_images.grad.sign()
        adv_images = adv_images + alpha * grad

        # Project the adversarial images back onto the epsilon ball and clip
        perturbation = torch.clamp(adv_images - images, -eps, eps)
        adv_images = torch.clamp(images + perturbation, 0, 1).detach()
        adv_images.requires_grad = True

        # if (_ + 1) % 10 == 0 or _ == 0:  # Log the first iteration and then every 10
        #     print(f"Iteration {_ + 1}/{n_iter}, Loss: {loss.item():.4f}")

    return adv_images


import matplotlib.pyplot as plt

def evaluate_impact_of_epsilon(model, dataloader, device):
    """
    Evaluate the impact of maximum perturbation (epsilon) on robust accuracy.

    Parameters:
    - model (torch.nn.Module): The trained model.
    - dataloader (DataLoader): DataLoader for the dataset to attack.
    - device (torch.device): Device to perform computations on.

    Returns:
    - None: Saves the plot to an output directory.
    """
    eps_values = [8 / 255, 16 / 255, 32 / 255, 64 / 255]
    alpha_values = [e / 10 for e in eps_values]
    robust_accuracies = []

    for eps, alpha in zip(eps_values, alpha_values):
        print(f"Evaluating for eps = {eps:.4f}, alpha = {alpha:.4f}")
        attack_params = {
            "eps": eps,
            "alpha": alpha,
            "n_iter": 50,
            "random_start": True
        }
        robust_accuracy, _ = evaluate_robust_accuracy(
            model=model,
            dataloader=dataloader,
            attack_fn=pgd_attack,
            attack_params=attack_params,
            device=device
        )
        robust_accuracies.append(robust_accuracy)
        print(f"Robust accuracy for eps {eps:.4f}: {robust_accuracy:.2f}%")

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(eps_values, robust_accuracies, marker="o", label="Robust Accuracy")
    plt.xlabel("Epsilon (Maximum Perturbation)")
    plt.ylabel("Robust Accuracy (%)")
    plt.title("Impact of Epsilon on Robust Accuracy")
    plt.grid(True)
    plt.xticks(eps_values, [f"{e:.4f}" for e in eps_values])
    plt.legend()
    plt.tight_layout()
    plt.savefig("./output/impact_of_epsilon.png")
    print("Plot saved to ./output/impact_of_epsilon.png")
