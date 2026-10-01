"""Reuse a library U-Net; no external weights or network access required."""
import segmentation_models_pytorch as smp


def build_model():
    return smp.Unet(encoder_name='resnet18', encoder_weights=None,
                    in_channels=3, classes=1, decoder_channels=(64, 32, 16, 8, 4))
