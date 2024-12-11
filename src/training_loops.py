import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset


def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    """
    Trains the model for one epoch.

    Parameters:
    - dataloader (DataLoader): DataLoader for the training dataset.
    - model (torch.nn.Module): Model to train.
    - loss_fn (torch.nn.Module): Loss function.
    - optimizer (torch.optim.Optimizer): Optimizer for updating parameters.
    - batch_size (int): Number of samples per batch.

    Returns:
    - float: Average loss for the epoch.
    """
    # Validate inputs
    if not isinstance(dataloader, DataLoader):
        raise ValueError("dataloader must be an instance of torch.utils.data.DataLoader")

    # Prepare model and device
    device = next(model.parameters()).device
    model.train()
    batch_losses = []  # Track losses for all batches

    # Iterate through mini-batches
    for batch, (X, y) in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset) / batch_size), desc="Training"):
        X, y = X.to(device), y.to(device)

        # Forward pass
        preds = model(X)
        loss = loss_fn(preds, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        # Optional gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Log loss for this batch
        batch_losses.append(loss.item())

    # Return the average loss for the epoch
    return np.mean(batch_losses)


def val_loop(dataloader, model, loss_fn, epoch_i, verbose=True):
    """
    Evaluate the model on the validation dataset.

    Parameters:
    - dataloader (DataLoader): DataLoader for the validation dataset.
    - model (torch.nn.Module): Model to evaluate.
    - loss_fn (torch.nn.Module): Loss function.
    - epoch_i (int): Current epoch number.
    - verbose (bool): Whether to print validation metrics.

    Returns:
    - float: Average validation loss.
    - float: Validation accuracy (percentage).
    """
    # Validate inputs
    if not isinstance(dataloader, DataLoader):
        raise ValueError("dataloader must be an instance of torch.utils.data.DataLoader")

    # Initialize evaluation
    device = next(model.parameters()).device
    model.eval()  # Set model to evaluation mode
    val_loss, num_correct = 0.0, 0
    total_samples = len(dataloader.dataset)
    num_batches = len(dataloader)

    # Evaluate in batches
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            preds = model(X)

            # Compute loss and accuracy
            val_loss += loss_fn(preds, y).item()
            num_correct += (preds.argmax(1) == y).type(torch.float).sum().item()

    # Compute average loss and accuracy
    val_loss /= num_batches
    accuracy = (num_correct / total_samples) * 100

    # Print metrics if verbose
    if verbose:
        print(f"Epoch {epoch_i}, Val Error: Accuracy: {accuracy:>0.1f}%, Avg loss: {val_loss:>8f}")

    return val_loss, accuracy


def train_model(model, x_train, y_train, x_val, y_val, optimizer, batch_size, loss_func, epochs):
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

    # Training and validation for each epoch
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        # Training loop
        train_loss = train_loop(train_loader, model, loss_func, optimizer, batch_size)
        print(f"Train Loss: {train_loss:.4f}")

        # Validation loop
        val_loss, val_accuracy = val_loop(val_loader, model, loss_func, epoch_i=epoch, verbose=False)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Log metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

    return history
