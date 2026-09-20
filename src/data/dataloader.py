from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataset import WasteDataset
from src.data.transforms import (
    get_train_transforms,
    get_eval_transforms,
)


def create_dataloaders(
    dataset_root,
    manifest_dir,
    batch_size=32,
    num_workers=0,
    pin_memory=True,
):
    """
    Create training and validation DataLoaders using the
    project's frozen dataset manifests.

    The test DataLoader is intentionally not created here.
    The test set remains isolated until final evaluation.
    """

    dataset_root = Path(dataset_root)
    manifest_dir = Path(manifest_dir)

    train_dataset = WasteDataset(
        manifest_path=manifest_dir / "train.csv",
        dataset_root=dataset_root,
        transform=get_train_transforms(),
    )

    val_dataset = WasteDataset(
        manifest_path=manifest_dir / "val.csv",
        dataset_root=dataset_root,
        transform=get_eval_transforms(),
        class_to_idx=train_dataset.class_to_idx,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "class_to_idx": train_dataset.class_to_idx,
    }