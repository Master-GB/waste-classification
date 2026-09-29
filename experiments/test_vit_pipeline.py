from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
import yaml

from src.data.dataloader import create_dataloaders
from src.models.vit import create_vit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "vit_b16.yaml"


def main():
    # Load configuration
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # Create dataloaders
    loaders = create_dataloaders(
        dataset_root=PROJECT_ROOT / config["data"]["dataset_root"],
        manifest_dir=PROJECT_ROOT / config["data"]["manifest_dir"],
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )

    train_loader = loaders["train_loader"]

    print(f"Training samples: {len(loaders['train_dataset'])}")
    print(f"Validation samples: {len(loaders['val_dataset'])}")

    # Create model
    model = create_vit(
        num_classes=10,
        pretrained=True,
        freeze_backbone=True,
    )

    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=0.001,
        weight_decay=0.0001,
    )

    # Get one batch
    images, labels = next(iter(train_loader))

    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")

    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)

    print(f"Output shape: {outputs.shape}")

    # Calculate loss
    loss = criterion(outputs, labels)

    print(f"Initial loss: {loss.item():.4f}")

    # Backward pass
    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

    print("Optimizer update successful.")
    print("ViT pipeline sanity check PASSED.")


if __name__ == "__main__":
    main()