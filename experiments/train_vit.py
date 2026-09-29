from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW

from src.data.dataloader import create_dataloaders
from src.models.vit import create_vit
from src.training.trainer import train_model
from src.utils.seed import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "vit_b16.yaml"


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    # Load configuration
    config = load_config(CONFIG_PATH)

    # Reproducibility
    seed = config["project"]["seed"] if "project" in config else config["experiment"]["seed"]
    set_seed(seed)

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Paths
    dataset_root = PROJECT_ROOT / config["data"]["dataset_root"]
    manifest_dir = PROJECT_ROOT / config["data"]["manifest_dir"]

    # Data
    loaders = create_dataloaders(
        dataset_root=dataset_root,
        manifest_dir=manifest_dir,
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"].get(
            "num_workers",
            config["training"].get("num_workers", 0),
        ),
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    class_to_idx = loaders["class_to_idx"]

    # Model
    model = create_vit(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        freeze_backbone=config["model"]["freeze_backbone"],
    )

    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Checkpoint path
    checkpoint_path = (
        PROJECT_ROOT
        / "results"
        / "vit_b16"
        / "best_model.pth"
    )

    print(f"Train samples: {len(loaders['train_dataset'])}")
    print(f"Validation samples: {len(loaders['val_dataset'])}")
    print(f"Number of classes: {len(class_to_idx)}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config["training"]["epochs"],
        checkpoint_path=checkpoint_path,
    )

    print("\nTraining completed.")
    print(f"Best checkpoint: {checkpoint_path}")

    return history


if __name__ == "__main__":
    main()