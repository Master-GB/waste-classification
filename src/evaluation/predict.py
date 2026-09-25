from pathlib import Path

import torch
from PIL import Image

from src.data.transforms import get_eval_transforms
from src.models.resnet50 import create_resnet50


CLASS_NAMES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
]


def load_resnet50_model(
    checkpoint_path,
    device,
):
    """
    Load the final fine-tuned ResNet50 model.
    """

    model = create_resnet50(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        freeze_backbone=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model


def predict_image(
    image_path,
    model,
    device,
):
    """
    Predict the waste class of a single image.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    transform = get_eval_transforms()

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = transform(image)

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():

        logits = model(image_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    confidence, predicted_index = (
        probabilities.max(dim=1)
    )

    predicted_class = CLASS_NAMES[
        predicted_index.item()
    ]

    return {
        "class": predicted_class,
        "confidence": confidence.item(),
        "probabilities": probabilities[
            0
        ].cpu().numpy(),
    }