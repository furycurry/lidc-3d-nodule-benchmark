"""
Faithful reimplementation of the 3D U-Net architecture from:
Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016).
"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation." MICCAI 2016.

Architecture matches the paper's Figure 2 exactly: 4 resolution levels,
channel-doubling-before-pooling scheme (to avoid bottlenecks, per Szegedy et al.),
two 3x3x3 convs + BatchNorm + ReLU per level, 2x2x2 max pooling / up-convolution.

Deliberate adaptations from the original paper (documented for reproducibility):
  1. 'Same' padding (padding=1) instead of valid convolutions, so output spatial
     size matches input size exactly (64^3 -> 64^3), rather than the paper's
     tiling-oriented shrinkage (132^3 -> 44^3).
  2. Single-channel sigmoid output (binary segmentation) instead of the paper's
     3-class softmax, to match this project's binary nodule/background masks.
  3. Standard BatchNorm3d behavior (running statistics at eval time) rather than
     the paper's current-batch-statistics-at-test-time workaround, since this
     project uses batch_size=8 rather than the paper's batch_size=1.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3x3 convolutions, each followed by BatchNorm3d then ReLU,
    exactly matching the paper's per-level analysis/synthesis pattern."""

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3DPaper(nn.Module):
    """
    3D U-Net (Çiçek et al., 2016), faithfully reimplemented.

    Channel progression follows Figure 2 exactly:
      Encoder: in->32->64 | 64->64->128 | 128->128->256 | 256->256->512 (bottleneck)
      Decoder: 512+256->256->256 | 256+128->128->128 | 128+64->64->64
      Final: 1x1x1 conv -> out_channels
    """

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # --- Analysis path (encoder) ---
        self.enc1 = ConvBlock(in_channels, 32, 64)
        self.enc2 = ConvBlock(64, 64, 128)
        self.enc3 = ConvBlock(128, 128, 256)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # --- Bottleneck ---
        self.bottleneck = ConvBlock(256, 256, 512)

        # --- Synthesis path (decoder) ---
        self.upconv3 = nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512 + 256, 256, 256)

        self.upconv2 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256 + 128, 128, 128)

        self.upconv1 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128 + 64, 64, 64)

        # --- Final 1x1x1 conv ---
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Analysis path, saving skip connections
        e1 = self.enc1(x)          # -> 64 channels
        p1 = self.pool(e1)

        e2 = self.enc2(p1)         # -> 128 channels
        p2 = self.pool(e2)

        e3 = self.enc3(p2)         # -> 256 channels
        p3 = self.pool(e3)

        b = self.bottleneck(p3)    # -> 512 channels

        # Synthesis path, with skip concatenation
        u3 = self.upconv3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.upconv2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.final_conv(d1)


if __name__ == "__main__":
    # Quick sanity check: confirm output shape matches input shape at 64^3
    model = UNet3DPaper(in_channels=1, out_channels=1)
    x = torch.randn(2, 1, 64, 64, 64)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {n_params:,}")