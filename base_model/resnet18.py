import torch
from torch import nn
from torch.nn import Conv2d
from torchvision import models
from torchvision.models import ResNet18_Weights


class BWResNet18(nn.Module):
    """ResNet18 classifier adapted for single-channel image inputs."""

    def __init__(
        self,
        n_classes: int,
        pretrained: bool = True,
        hid_lay_size: int = 100,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the classification backbone and output head."""
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        pretrained_conv1_weight = None
        if pretrained:
            pretrained_conv1_weight = backbone.conv1.weight.detach().mean(dim=1, keepdim=True)

        backbone.conv1 = Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained_conv1_weight is not None:
            with torch.no_grad():
                backbone.conv1.weight.copy_(pretrained_conv1_weight)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, hid_lay_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_lay_size, n_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return class logits."""
        return self.backbone(x)
