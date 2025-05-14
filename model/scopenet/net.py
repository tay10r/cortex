from torch import nn, Tensor, concat
from torch.nn import functional as F

from scopenet.config import NetConfig


class _SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        w = self.fc(self.pool(x))
        return x * w


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels,
                      channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels,
                      channels,
                      kernel_size=3,
                      padding=1,
                      bias=False)
        )

    def forward(self, x):
        return F.leaky_relu(x + self.block(x))


def make_res_blocks(channels: int, n: int) -> nn.Sequential:
    return nn.Sequential(*[_ResBlock(channels) for _ in range(n)])


class _Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 se_enabled: bool,
                 res_blocks: int,
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
        self.se = _SEBlock(out_channels) if se_enabled else nn.Identity()
        self.res_blocks = make_res_blocks(out_channels, res_blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = F.leaky_relu(x)
        x = self.se(x)
        x = self.res_blocks(x)
        return x


class _Decoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 se_enabled: bool,
                 res_blocks: int):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels,
                                         out_channels,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1)
        self.se = _SEBlock(out_channels) if se_enabled else nn.Identity()
        self.res_blocks = make_res_blocks(out_channels, res_blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_t(x)
        x = F.leaky_relu(x)
        x = self.se(x)
        x = self.res_blocks(x)
        return x


class Net(nn.Module):
    def __init__(self, in_channels: int, config: NetConfig):
        super().__init__()

        # shorthand for the weight multiplier
        q = config.weight_multiplier

        self.enc1 = _Encoder(in_channels,
                             8 * q,
                             stride=2,
                             se_enabled=config.se_enabled,
                             res_blocks=config.num_encoder_res_blocks)
        self.enc2 = _Encoder(8 * q,
                             16 * q,
                             stride=2,
                             se_enabled=config.se_enabled,
                             res_blocks=config.num_encoder_res_blocks)
        self.enc3 = _Encoder(16 * q,
                             32 * q,
                             stride=2,
                             se_enabled=config.se_enabled,
                             res_blocks=config.num_encoder_res_blocks)
        self.enc4 = _Encoder(32 * q,
                             64 * q,
                             stride=2,
                             se_enabled=config.se_enabled,
                             res_blocks=config.num_encoder_res_blocks)
        self.enc5 = _Encoder(64 * q,
                             128 * q,
                             stride=2,
                             se_enabled=config.se_enabled,
                             res_blocks=config.num_encoder_res_blocks)
        self.res_blocks = make_res_blocks(128 * q,
                                          config.num_bottleneck_res_blocks)
        self.dec5 = _Decoder(128 * q,
                             64 * q,
                             se_enabled=config.se_enabled,
                             res_blocks=config.num_decoder_res_blocks)
        self.dec4 = _Decoder(128 * q,
                             32 * q,
                             se_enabled=config.se_enabled,
                             res_blocks=config.num_decoder_res_blocks)
        self.dec3 = _Decoder(64 * q,
                             16 * q,
                             se_enabled=config.se_enabled,
                             res_blocks=config.num_decoder_res_blocks)
        self.dec2 = _Decoder(32 * q,
                             8 * q,
                             se_enabled=config.se_enabled,
                             res_blocks=config.num_decoder_res_blocks)
        self.dec1 = _Decoder(16 * q,
                             8 * q,
                             se_enabled=config.se_enabled,
                             res_blocks=config.num_decoder_res_blocks)
        self.last = nn.Sequential(
            _ResBlock(8 * q),
            _ResBlock(8 * q),
            _ResBlock(8 * q),
            _ResBlock(8 * q),
            nn.Conv2d(8 * q, in_channels, kernel_size=3,
                      padding=1, bias=False)
        )
        if config.final_activation:
            self.last.add_module(nn.Sigmoid())

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
        return self.last(d1)
