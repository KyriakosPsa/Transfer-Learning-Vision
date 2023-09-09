# %% [code] {"papermill":{"duration":5.783081,"end_time":"2023-09-08T08:14:26.616444","exception":false,"start_time":"2023-09-08T08:14:20.833363","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-09-09T08:43:40.255315Z","iopub.execute_input":"2023-09-09T08:43:40.255849Z","iopub.status.idle":"2023-09-09T08:43:40.283101Z","shell.execute_reply.started":"2023-09-09T08:43:40.255790Z","shell.execute_reply":"2023-09-09T08:43:40.282318Z"}}
"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, f1_score
from collections import deque  # Import deque for early stopping
import warnings

# Suppress the specific UserWarning related to y_pred and y_true class mismatch
warnings.filterwarnings("ignore", category=UserWarning, message="y_pred contains classes not in y_true")


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_bal_acc, train_mcc, train_f_score = 0, 0, 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
#         train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        # get the balanced accuracy
        train_bal_acc += balanced_accuracy_score(y.cpu().numpy(), y_pred_class.cpu().numpy())
        train_mcc += matthews_corrcoef(y.cpu().numpy(), y_pred_class.cpu().numpy())
        train_f_score += f1_score(y.cpu().numpy(), y_pred_class.cpu().numpy(), average='weighted')

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
#     train_acc = train_acc / len(dataloader)
    train_bal_acc = train_bal_acc / len(dataloader)
    train_mcc = train_mcc / len(dataloader)
    train_f_score = train_f_score / len(dataloader)
    return train_loss, train_bal_acc, train_mcc, train_f_score


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_bal_acc, test_mcc, test_f_score = 0, 0, 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
#             test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
            # get the balanced accuracy
            test_bal_acc += balanced_accuracy_score(y.cpu().numpy(), test_pred_labels.cpu().numpy())
            test_mcc += matthews_corrcoef(y.cpu().numpy(), test_pred_labels.cpu().numpy())
            test_f_score += f1_score(y.cpu().numpy(), test_pred_labels.cpu().numpy(), average='weighted')

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
#     test_acc = test_acc / len(dataloader)
    test_bal_acc = test_bal_acc / len(dataloader)
    test_mcc = test_mcc / len(dataloader)
    test_f_score = test_f_score / len(dataloader)
    return test_loss, test_bal_acc, test_mcc, test_f_score



def train_with_early_stopping(model: torch.nn.Module,
                              train_dataloader: torch.utils.data.DataLoader,
                              valid_dataloader: torch.utils.data.DataLoader,
                              optimizer: torch.optim.Optimizer,
                              loss_fn: torch.nn.Module,
                              epochs: int,
                              device: torch.device,
                              patience: int = 5) -> Dict[str, List]:
    """Trains and tests a PyTorch model with early stopping.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    valid_dataloader: A DataLoader instance for the model to be validated on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g., "cuda" or "cpu").
    patience: An integer indicating the number of epochs to wait for improvement
              before early stopping (default is 5).

    Returns:
    A dictionary of training and validation loss as well as training and
    validation metrics. Each metric has a value in a list for each epoch.
    """
    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_bal_acc": [],
        "train_mcc": [],
        "train_f_score": [],
        "valid_loss": [],
        "valid_bal_acc": [],
        "valid_mcc": [],
        "valid_f_score": [],
    }

    # Initialize variables for early stopping
    best_valid_loss = float('inf')
    no_improvement_count = 0
    best_model_weights = model.state_dict()

    # Create a deque to keep track of the validation loss history
    validation_loss_history = deque(maxlen=patience)

    # Make sure model is on the target device
    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_bal_acc, train_mcc, train_f_score = train_step(model=model,
                                                                         dataloader=train_dataloader,
                                                                         loss_fn=loss_fn,
                                                                         optimizer=optimizer,
                                                                         device=device)
        valid_loss, valid_bal_acc, valid_mcc, valid_f_score = test_step(model=model,
                                                                       dataloader=valid_dataloader,
                                                                       loss_fn=loss_fn,
                                                                       device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_bal_acc: {train_bal_acc:.4f} | "
            f"train_mcc: {train_mcc:.4f} | "
            f"valid_loss: {valid_loss:.4f} | "
            f"valid_bal_acc: {valid_bal_acc:.4f} | "
            f"valid_mcc: {valid_mcc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_bal_acc"].append(train_bal_acc)
        results["train_mcc"].append(train_mcc)
        results["train_f_score"].append(train_f_score)
        results["valid_loss"].append(valid_loss)
        results["valid_bal_acc"].append(valid_bal_acc)
        results["valid_mcc"].append(valid_mcc)
        results["valid_f_score"].append(valid_f_score)

        # Append the validation loss to the history
        validation_loss_history.append(valid_loss)

        # Check if the validation loss improved
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            no_improvement_count = 0
            # Save the best model weights
            best_model_weights = model.state_dict()
        else:
            no_improvement_count += 1

        # Check if early stopping criteria are met
        if no_improvement_count >= patience:
            print(f"Early stopping after {epoch + 1} epochs without improvement.")
            break

    # Load the best model weights
    model.load_state_dict(best_model_weights)

    # Return the filled results and best model
    return results, model