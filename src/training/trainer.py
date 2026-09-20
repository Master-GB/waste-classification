import time

import torch
from tqdm.auto import tqdm
from pathlib import Path



def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    """
    Train a model for one epoch.

    Returns:
        average_loss
        accuracy
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False,
    )

    for images, labels in progress_bar:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        # Clear gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Statistics
        batch_size = images.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
):
    """
    Evaluate a model on validation data.

    No gradients or parameter updates are performed.
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():

        for images, labels in tqdm(
            dataloader,
            desc="Validation",
            leave=False,
        ):

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            batch_size = images.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
):
    """
    Evaluate a model on validation data.

    No gradients or parameter updates are performed.
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():

        for images, labels in tqdm(
            dataloader,
            desc="Validation",
            leave=False,
        ):

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            batch_size = images.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    scheduler=None,
    checkpoint_path=None,
):
    """
    Train and validate a model for multiple epochs.

    The best model is selected using validation loss.
    """

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_time_seconds": [],
    }

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):

        print(f"\nEpoch {epoch}/{epochs}")

        start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.perf_counter() - start_time

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["epoch_time_seconds"].append(epoch_time)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Val Acc:   {val_accuracy:.4f}"
        )

        print(f"Time: {epoch_time:.1f}s")

        # Save best validation model
        if (
            checkpoint_path is not None
            and val_loss < best_val_loss
        ):
            best_val_loss = val_loss

            checkpoint_path = Path(checkpoint_path)

            checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path,
            )

            print(
                f"Best model saved "
                f"(val_loss={val_loss:.4f})"
            )

    return history