from __future__ import annotations

import torch
from torch import nn


class TorchvisionSegmentationWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


DEFAULT_MODEL = "deeplabv3plus_efficientnet_b4"


def build_model(model_name: str = DEFAULT_MODEL) -> nn.Module:
    model_name = model_name.lower()

    if model_name in {"unet_efficientnet_b4", "deeplabv3plus_efficientnet_b4", "unetplusplus_efficientnet_b4"}:
        try:
            import segmentation_models_pytorch as smp
        except ImportError as error:
            raise RuntimeError(
                "Install segmentation-models-pytorch to use the instrument segmentation models: "
                "`python -m pip install segmentation-models-pytorch`."
            ) from error

        if model_name == "unet_efficientnet_b4":
            return smp.Unet(
                encoder_name="efficientnet-b4",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            )
        if model_name == "deeplabv3plus_efficientnet_b4":
            return smp.DeepLabV3Plus(
                encoder_name="efficientnet-b4",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            )
        return smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

    if model_name == "torchvision_deeplabv3_resnet50":
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
        return TorchvisionSegmentationWrapper(model)

    raise ValueError(
        "Unknown model. Use one of: unet_efficientnet_b4, "
        "unetplusplus_efficientnet_b4, deeplabv3plus_efficientnet_b4, "
        "torchvision_deeplabv3_resnet50."
    )
