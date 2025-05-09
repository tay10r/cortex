from torch import nn, Tensor, concat
from torch.nn import functional as F


class _Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.norm(self.conv(x)))


class _Decoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              padding=1,
                              bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=False)
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels,
                      channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels,
                      channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


class Net(nn.Module):
    def __init__(self, in_channels: int = 3, q: int = 2):
        super().__init__()
        self.enc1 = _Encoder(in_channels, 8 * q, stride=2)
        self.enc2 = _Encoder(8 * q, 16 * q, stride=2)
        self.enc3 = _Encoder(16 * q, 32 * q, stride=2)
        self.enc4 = _Encoder(32 * q, 64 * q, stride=2)
        self.enc5 = _Encoder(64 * q, 128 * q, stride=2)
        self.res_blocks = nn.Sequential(
            _ResBlock(128 * q),
            _ResBlock(128 * q),
            _ResBlock(128 * q),
            _ResBlock(128 * q)
        )
        self.dec5 = _Decoder(128 * q, 64 * q)
        self.dec4 = _Decoder(128 * q, 32 * q)
        self.dec3 = _Decoder(64 * q, 16 * q)
        self.dec2 = _Decoder(32 * q, 8 * q)
        self.dec1 = _Decoder(16 * q, 8 * q)
        self.refine = nn.Sequential(
            nn.Conv2d(8 * q, 8 * q, kernel_size=3,
                      padding=1, bias=False),
            nn.Conv2d(8 * q, in_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e5 = self.res_blocks(e5)
        d5 = self.dec5(e5)
        d4 = self.dec4(concat((e4, d5), dim=1))
        d3 = self.dec3(concat((e3, d4), dim=1))
        d2 = self.dec2(concat((e2, d3), dim=1))
        d1 = self.dec1(concat((e1, d2), dim=1))
        return x + self.refine(d1)
