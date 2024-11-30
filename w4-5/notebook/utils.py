"""Utils to get rid of low level stuff."""

from math import inf

import keras
import torch
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cifar_dataloaders(
    batch_size,
    transform=None,
    validation=True,
    validation_percent=0.1,
):

    if transform is None:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.PILToTensor(),
                transforms.Lambda(lambda x: x / 255.0),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.PILToTensor(), transforms.Lambda(lambda x: x / 255.0)]
        )
    else:
        train_transform = transform
        test_transform = transform

    train_set = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    if validation:
        train_set, validation_set = torch.utils.data.random_split(
            train_set, (1 - validation_percent, validation_percent)
        )
    test_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    if validation:
        validation_loader = torch.utils.data.DataLoader(
            validation_set, batch_size=batch_size, shuffle=False
        )
        return train_loader, validation_loader, test_loader

    return train_loader, test_loader


def train(
    dataloader, model, loss_fn, optimizer, epochs, validation_loader=None, patience=5
):

    min_val_loss = inf
    best_state = None
    patience_count = 0

    history = {
        "acc": [],
        "prec": [],
        "rec": [],
        "f1": [],
        "loss": [],
        "val_acc": [],
        "val_prec": [],
        "val_rec": [],
        "val_f1": [],
        "val_loss": [],
    }

    for epoch in range(epochs):

        acc_metric = MulticlassAccuracy().to(device)
        prec_metric = MulticlassPrecision().to(device)
        rec_metric = MulticlassRecall().to(device)
        f1_metric = MulticlassF1Score().to(device)

        model.train()

        print(f"Epoch {epoch+1}/{epochs}")

        # Set the target of the pbar to
        # the number of iterations (batches) in the dataloader
        n_batches = len(dataloader)
        pbar = keras.utils.Progbar(target=n_batches)

        total_loss = 0

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            # Compute prediction and loss
            optimizer.zero_grad()

            pred = model(X)
            pred = pred.squeeze()

            y = torch.nn.functional.one_hot(y, num_classes=10)
            y = y.squeeze()
            y = y.type(torch.float32)

            loss = loss_fn(pred, y)
            total_loss += loss.item()

            pred, y = torch.argmax(pred, dim=1), torch.argmax(y, dim=1)

            acc_metric.update(pred, y)
            acc = acc_metric.compute().cpu()

            prec_metric.update(pred, y)
            prec = prec_metric.compute().cpu()

            rec_metric.update(pred, y)
            rec = rec_metric.compute().cpu()

            f1_metric.update(pred, y)
            f1 = f1_metric.compute().cpu()

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update the pbar with each batch
            pbar.update(
                batch,
                values=[
                    ("loss", total_loss / (batch + 1)),
                    ("acc", acc),
                    ("precision", prec),
                    ("recall", rec),
                    ("f1", f1),
                ],
            )

            # Finish the progress bar with a final update
            # This ensures the progress bar reaches the end
            # (i.e. the target of n_batches is met on the last batch)
            if batch + 1 == n_batches:
                pbar.update(n_batches, values=None)

        history["acc"].append(acc)
        history["prec"].append(prec)
        history["rec"].append(rec)
        history["f1"].append(f1)
        history["loss"].append(total_loss / (batch + 1))

        if validation_loader is None:
            continue

        model.eval()
        print(f"Validation {epoch+1}/{epochs}")

        n_batches = len(validation_loader)
        pbar = keras.utils.Progbar(target=n_batches)

        val_acc_metric = MulticlassAccuracy().to(device)
        val_prec_metric = MulticlassPrecision().to(device)
        val_rec_metric = MulticlassRecall().to(device)
        val_f1_metric = MulticlassF1Score().to(device)

        total_val_loss = 0

        for batch, (X, y) in enumerate(validation_loader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            pred = pred.squeeze()

            y = torch.nn.functional.one_hot(y, num_classes=10)
            y = y.squeeze()
            y = y.type(torch.float32)

            val_loss = loss_fn(pred, y)
            total_val_loss += val_loss.item()

            pred, y = torch.argmax(pred, dim=1), torch.argmax(y, dim=1)

            val_acc_metric.update(pred, y)
            val_acc = acc_metric.compute().cpu()

            val_prec_metric.update(pred, y)
            val_prec = prec_metric.compute().cpu()

            val_rec_metric.update(pred, y)
            val_rec = rec_metric.compute().cpu()

            val_f1_metric.update(pred, y)
            val_f1 = f1_metric.compute().cpu()

            # Update the pbar with each batch
            pbar.update(
                batch,
                values=[
                    ("val_loss", total_val_loss / (batch + 1)),
                    ("val_acc", val_acc),
                    ("val_precision", val_prec),
                    ("val_recall", val_rec),
                    ("val_f1", val_f1),
                ],
            )

            # Finish the progress bar with a final update
            # This ensures the progress bar reaches the end
            # (i.e. the target of n_batches is met on the last batch)
            if batch + 1 == n_batches:
                pbar.update(n_batches, values=None)

        total_val_loss = total_val_loss / (batch + 1)

        history["val_acc"].append(val_acc)
        history["val_prec"].append(val_prec)
        history["val_rec"].append(val_rec)
        history["val_f1"].append(val_f1)
        history["val_loss"].append(total_val_loss)

        if total_val_loss <= min_val_loss:
            min_val_loss = total_val_loss
            patience_count = 0
            best_state = model.state_dict()
        else:
            patience_count += 1

        if patience_count >= patience:
            model.load_state_dict(best_state)
            print("Training stopped.")
            break

    return model, history


def test(dataloader, model, loss_fn):

    print("Test")

    acc_metric = MulticlassAccuracy().to(device)
    prec_metric = MulticlassPrecision().to(device)
    rec_metric = MulticlassRecall().to(device)
    f1_metric = MulticlassF1Score().to(device)

    model.eval()

    n_batches = len(dataloader)
    pbar = keras.utils.Progbar(target=n_batches)

    total_loss = 0

    preds, ys = [], []

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        pred = pred.squeeze()

        ys += y.tolist()
        preds += pred.tolist()

        y = torch.nn.functional.one_hot(y, num_classes=10)
        y = y.squeeze()
        y = y.type(torch.float32)

        loss = loss_fn(pred, y)
        total_loss += loss.item()

        pred, y = torch.argmax(pred, dim=1), torch.argmax(y, dim=1)

        acc_metric.update(pred, y)
        acc = acc_metric.compute().cpu()

        prec_metric.update(pred, y)
        prec = prec_metric.compute().cpu()

        rec_metric.update(pred, y)
        rec = rec_metric.compute().cpu()

        f1_metric.update(pred, y)
        f1 = f1_metric.compute().cpu()

        # Update the pbar with each batch
        pbar.update(
            batch,
            values=[
                ("loss", total_loss / (batch + 1)),
                ("acc", acc),
                ("precision", prec),
                ("recall", rec),
                ("f1", f1),
            ],
        )

        # Finish the progress bar with a final update
        # This ensures the progress bar reaches the end
        # (i.e. the target of n_batches is met on the last batch)
        if batch + 1 == n_batches:
            pbar.update(n_batches, values=None)

    metrics = {
        "loss",
        total_loss / (batch + 1),
        ("val_acc", acc),
        ("precision", prec),
        ("recall", rec),
        ("f1", f1),
    }

    return preds, ys, metrics
