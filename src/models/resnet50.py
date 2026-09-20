import torch.nn as nn

from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)


def create_resnet50(
    num_classes=10,
    pretrained=True,
    freeze_backbone=True,
):
    """
    Create a ResNet50 model for waste classification.

    Args:
        num_classes:
            Number of output classes.

        pretrained:
            If True, use ImageNet pretrained weights.

        freeze_backbone:
            If True, freeze the ResNet50 feature extractor
            and train only the final classification layer.

    Returns:
        model
    """

    if pretrained:
        weights = ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = resnet50(weights=weights)

    # Freeze pretrained backbone
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    # Replace original ImageNet classifier
    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes,
    )

    return model