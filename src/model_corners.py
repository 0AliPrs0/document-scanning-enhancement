import torch
import torch.nn as nn


class CornerDirectRegressor(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3) if use_dropout else nn.Identity(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.3) if use_dropout else nn.Identity(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 8),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Input: (B, 3, H, W) -> Output: (B, 8) normalized corner coordinates
        x = self.features(x)
        return self.regressor(x)


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


class CornerHeatmapUNet(nn.Module):
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

        self.outc = nn.Conv2d(128, 4, kernel_size=1)

    def forward(self, x):
        # Encoder: progressively downsamples spatial features.
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder: upsamples and uses skip connections from the encoder.
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        # Output: 4 heatmaps, one for each document corner.
        return self.outc(x)
