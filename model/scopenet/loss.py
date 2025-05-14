import torch
from torch import Tensor, nn
import torch.nn.functional as F


def gaussian_window(window_size: int, sigma: float, device):
    coords = torch.arange(window_size, dtype=torch.float32,
                          device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.size_average = size_average

    def forward(self, img1: Tensor, img2: Tensor):
        # Assume shape is (B, C, H, W)
        C = img1.size(1)
        window = gaussian_window(self.window_size, self.sigma, img1.device)
        window = window.expand(C, 1, self.window_size, self.window_size)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=C)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=C)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window,
                             padding=self.window_size // 2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window,
                             padding=self.window_size // 2, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window,
                           padding=self.window_size // 2, groups=C) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(dim=(1, 2, 3))


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()

        def conv_block(in_ch, out_ch, norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            conv_block(in_channels, 8, norm=False),
            conv_block(8, 16),
            conv_block(16, 32),
            conv_block(32, 64),
            nn.Conv2d(64, 1, 4, padding=1)  # Patch output
        )

    def forward(self, x: Tensor):
        return self.model(x)


def bce_loss(logits, target_is_real):
    targets = torch.ones_like(
        logits) if target_is_real else torch.zeros_like(logits)
    return F.binary_cross_entropy_with_logits(logits, targets)


loss_functions = {
    'ssim': SSIMLoss(),
    'l1': nn.L1Loss(),
    'smooth_l1': nn.SmoothL1Loss(),
    'mse': nn.MSELoss()
}
