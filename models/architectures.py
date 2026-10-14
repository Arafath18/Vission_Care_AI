"""Neural network architectures — CPU optimized."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import monai.networks.nets as monai_nets


def _replace_classifier(model: nn.Module, num_classes: int, in_features: int) -> nn.Module:
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 Classifier matching trained checkpoint."""
    def __init__(self, num_classes: int = 11, pretrained: bool = True):
        super().__init__()
        weights = (
            models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = models.efficientnet_b0(weights=weights)
        in_f = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_f, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)


class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights = (
            models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = models.mobilenet_v3_small(weights=weights)
        in_f = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_f, 256),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.backbone.classifier[0](x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        return x * self.psi(psi)


class AttentionUNet(nn.Module):
    """MONAI-based Attention U-Net for retinal lesion segmentation."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.net = monai_nets.AttentionUnet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(32, 64, 128),
            strides=(2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepLabV3Plus(nn.Module):
    """DeepLabV3++ (DeepLabV3Plus) architecture for lesion boundary detection."""
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        # Backbone (lightweight features encoder)
        self.backbone_low = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.backbone_high = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # ASPP Block
        self.aspp1 = nn.Conv2d(64, 32, 1)
        self.aspp2 = nn.Conv2d(64, 32, 3, padding=2, dilation=2)
        self.aspp3 = nn.Conv2d(64, 32, 3, padding=4, dilation=4)
        self.aspp_conv = nn.Sequential(
            nn.Conv2d(96, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.low_conv = nn.Sequential(
            nn.Conv2d(16, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(80, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_features = self.backbone_low(x)
        high_features = self.backbone_high(low_features)
        
        a1 = self.aspp1(high_features)
        a2 = self.aspp2(high_features)
        a3 = self.aspp3(high_features)
        aspp_out = self.aspp_conv(torch.cat([a1, a2, a3], dim=1))
        
        aspp_out_up = F.interpolate(aspp_out, size=low_features.shape[2:], mode='bilinear', align_corners=True)
        
        low_feat_conv = self.low_conv(low_features)
        
        out = self.decoder(torch.cat([aspp_out_up, low_feat_conv], dim=1))
        return out
