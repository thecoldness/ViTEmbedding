
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
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
    model.train()

    train_loss, train_recall, train_precision, train_F1Score = 0, 0, 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss, TP , FP ,FN = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        train_precision += precision.mean().item()
        train_recall += recall.mean().item()
        train_F1Score += (2 * (precision + recall) / (precision + recall)).mean().item()


    train_loss = train_loss / len(dataloader)
    train_precision = train_precision / len(dataloader)
    train_recall = train_recall / len(dataloader)
    train_F1Score = train_F1Score / len(dataloader)


    return train_loss, train_precision , train_recall , train_F1Score


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
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
    model.eval()

    test_loss, test_precision, test_recall, test_F1Score = 0, 0, 0 , 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss, TP , FP , FN = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            test_precision += precision.mean().item()
            test_recall += recall.mean().item()
            test_F1Score += (2 * (precision + recall) / (precision + recall)).mean().item()



        test_loss = test_loss / len(dataloader)
        test_precision = test_precision / len(dataloader)
        test_recall = test_recall / len(dataloader)
        test_F1Score = test_F1Score / len(dataloader)

        return test_loss, test_precision , test_recall , test_F1Score


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]}
    For example if training for epochs=2:
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]}
    """
    results = {"train_loss": [],
               "train_precision": [],
               "train_recall" : [],
               "train_F1Score" : [],
               "test_loss": [],
               "test_precision": [],
               "test_recall":[],
               "test_F1Score":[]
               }
    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_precision, train_recall, train_F1Score = train_step(model=model,
                                                                              dataloader=train_dataloader,
                                                                              loss_fn=loss_fn,
                                                                              optimizer=optimizer,
                                                                              device=device)
        test_loss, test_precision, test_recall, test_F1Score = test_step(model=model,
                                                                           dataloader=test_dataloader,
                                                                           loss_fn=loss_fn,
                                         
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_precision: {train_precision:.4f} | "
            f"train_recall: {train_recall:.4f} | "
            f"train_F1Score: {train_F1Score:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_precision: {test_precision:.4f} | "
            f"train_recall: {test_recall:.4f} | "
            f"train_F1Score: {test_F1Score:.4f} | "
        )

        results["train_loss"].append(train_loss)
        results["train_precision"].append(train_precision)
        results["train_recall"].append(train_recall)
        results["train_F1Score"].append(train_F1Score)
        results["test_loss"].append(test_loss)
        results["test_precision"].append(test_precision)
        results["test_recall"].append(test_recall)
        results["test_F1Score"].append(test_F1Score)

    return results
