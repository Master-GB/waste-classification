"""EfficientNetB0 for the team's waste classes.

Uses torchvision ImageNet weights and the team's existing trainer. The wrapper
keeps a frozen feature extractor in evaluation mode when that trainer calls
model.train(), so BatchNorm running statistics do not change accidentally.
"""

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class WasteEfficientNetB0(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, freeze_backbone=True):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.network = efficientnet_b0(weights=weights)
        in_features = self.network.classifier[1].in_features
        self.network.classifier[1] = nn.Linear(in_features, num_classes)
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable):
        """Call before creating a new optimizer when starting fine-tuning."""
        self.freeze_backbone = not trainable
        for parameter in self.network.features.parameters():
            parameter.requires_grad = trainable
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.network.features.eval()
        return self

    def forward(self, images):
        # Raw logits: CrossEntropyLoss applies its own log-softmax.
        return self.network(images)


def create_efficientnetb0(num_classes=10, pretrained=True, freeze_backbone=True):
    return WasteEfficientNetB0(num_classes, pretrained, freeze_backbone)
