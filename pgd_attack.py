import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def pgd_attack(
    model, 
    images, 
    labels, 
    eps=32/255, 
    alpha=(32/255)/10, 
    n_iter=50, 
    random_start=True
):
    """
    Perform the Projected Gradient Descent (PGD) attack.

    Parameters:
    - model (torch.nn.Module): The model to attack.
    - images (torch.Tensor): Input images (normalized).
    - labels (torch.Tensor): Ground truth labels for the input images.
    - eps (float): Maximum perturbation (epsilon).
    - alpha (float): Step size for each iteration.
    - n_iter (int): Number of attack iterations.
    - random_start (bool): Whether to initialize adversarial examples randomly within the epsilon ball.

    Returns:
    - torch.Tensor: Adversarial examples.
    """
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

    return adv_images
