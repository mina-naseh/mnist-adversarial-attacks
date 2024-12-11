from tqdm import tqdm
import torch
from src.pgd_attack import pgd_attack
from src.training_loops import val_loop
from torch.utils.data import DataLoader, TensorDataset


def train_loop_robust(dataloader, model, loss_fn, optimizer, batch_size, attack_fn=None, attack_params=None):
    """
    Trains the model for one epoch with optional adversarial training.

    Parameters:
    - dataloader (DataLoader): DataLoader for the training dataset.
    - model (torch.nn.Module): Model to train.
    - loss_fn (torch.nn.Module): Loss function.
    - optimizer (torch.optim.Optimizer): Optimizer for updating parameters.
    - batch_size (int): Number of samples per batch.
    - attack_fn (callable, optional): Function to generate adversarial examples.
    - attack_params (dict, optional): Parameters for the attack function.

    Returns:
    - float: Average loss for the epoch.
    """
    device = next(model.parameters()).device
    model.train()
    batch_losses = []

    for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
        X, y = X.to(device), y.to(device)

        # Adversarial training: 3/4 clean, 1/4 adversarial
        clean_size = int(3 * batch_size / 4)
        adv_size = batch_size - clean_size

        X_clean, y_clean = X[:clean_size], y[:clean_size]
        X_adv, y_adv = X[clean_size:clean_size + adv_size], y[clean_size:clean_size + adv_size]

        if attack_fn is not None:
            model.eval()
            X_adv = attack_fn(model, X_adv, y_adv, **attack_params)
            model.train()

        # Combine clean and adversarial examples
        X_combined = torch.cat([X_clean, X_adv], dim=0)
        y_combined = torch.cat([y_clean, y_adv], dim=0)

        # Compute predictions and loss
        preds = model(X_combined)
        loss = loss_fn(preds, y_combined)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        batch_losses.append(loss.item())

    # Return the average loss for the epoch
    return sum(batch_losses) / len(batch_losses)


def train_model_robust(
    model, x_train, y_train, x_val, y_val, optimizer, batch_size, loss_func, epochs, attack_fn=None, attack_params=None
):
    """
    Train the model and evaluate on validation data after each epoch.

    Parameters:
    - model (torch.nn.Module): The model to train.
    - x_train, y_train (torch.Tensor): Training features and labels.
    - x_val, y_val (torch.Tensor): Validation features and labels.
    - optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
    - batch_size (int): Batch size for training.
    - loss_func (torch.nn.Module): Loss function.
    - epochs (int): Number of epochs to train.
    - attack_fn (callable, optional): Function to generate adversarial examples.
    - attack_params (dict, optional): Parameters for the attack function.

    Returns:
    - dict: Dictionary containing training and validation metrics.
    """
    # Prepare data loaders
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Track metrics
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        # Training loop with adversarial examples
        train_loss = train_loop_robust(train_loader, model, loss_func, optimizer, batch_size, attack_fn, attack_params)
        print(f"Train Loss: {train_loss:.4f}")

        # Validation loop
        val_loss, val_accuracy = val_loop(val_loader, model, loss_func, epoch_i=epoch, verbose=True)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Log metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

    return history

