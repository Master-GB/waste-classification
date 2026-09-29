import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


def create_vit(num_classes=10, pretrained=True, freeze_backbone=True):
    if pretrained:
        weights = ViT_B_16_Weights.DEFAULT
    else:
        weights = None

    model = vit_b_16(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    return model