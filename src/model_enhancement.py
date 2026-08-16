import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DocumentEnhancementUNet(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()

        self.inc = DoubleConv(3, 64)

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128),
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256),
            nn.Dropout(0.2) if use_dropout else nn.Identity(),
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512),
            nn.Dropout(0.3) if use_dropout else nn.Identity(),
        )

        self.up1 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2,
        )

        self.conv_up1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2,
        )

        self.conv_up2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2,
        )

        self.conv_up3 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Encoder: 3-channel input -> progressively deeper feature maps.
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder: restores spatial resolution using encoder skip connections.
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)

        # Output: 3-channel restored image in [0, 1].
        return torch.sigmoid(self.outc(x))