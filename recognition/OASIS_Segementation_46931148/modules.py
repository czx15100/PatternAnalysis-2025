"""
U-Net Model for MRI Brain Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Convolutional layers with ReLU, normalization and dropout.
    """
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass for DoubleConv with residual connection."""
        out = self.conv(x)
        x = self.skip(x)
        out = out + x
        return self.relu(out)


class Down(nn.Module):
    """Downsampling with maxpool followed by DoubleConv."""
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout)
        )

    def forward(self, x):
        """Forward pass for Down block."""
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsampling with transposed conv followed by DoubleConv."""
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout)

    def forward(self, x, skip):
        """Forward pass for Up block with skip connection."""
        x = self.up(x)

        # Adjust size mismatch
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(
            x,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )

        # Concatenate skip connection
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture for 2D image segmentation."""
    def __init__(self, in_channels=1, num_classes=3, dropout=0.1):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(in_channels, 64, dropout)
        self.down1 = Down(64, 128, dropout)
        self.down2 = Down(128, 256, dropout)
        self.down3 = Down(256, 512, dropout)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024, dropout)

        # Decoder
        self.up3 = Up(1024, 512, dropout)
        self.up2 = Up(512, 256, dropout)
        self.up1 = Up(256, 128, dropout)
        self.up0 = Up(128, 64, dropout)

        # Final output layer
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass through the full U-Net."""
         # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        x5 = self.bottleneck(x4)

        # Decoder
        x = self.up3(x5, x4)
        x = self.up2(x, x3)
        x = self.up1(x, x2)
        x = self.up0(x, x1)

        return self.outc(x)