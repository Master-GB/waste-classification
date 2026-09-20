from pathlib import Path

import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    val_accuracy,
    class_to_idx,
    save_path,
):
    """
    Save model and training state.
    """

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "class_to_idx": class_to_idx,
    }

    torch.save(
        checkpoint,
        save_path,
    )


    