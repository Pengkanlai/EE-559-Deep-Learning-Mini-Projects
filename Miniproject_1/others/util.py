import torch


def compute_psnr(denoised, target):
    denoised = denoised.float().div(255.0)
    target = target.float().div(255.0)
    mse = torch.mean((denoised - target) ** 2)

    return -10 * torch.log10(mse + 10 ** -8)


def test_compute_psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x - y) ** 2).mean((1, 2, 3))).mean()


if __name__ == '__main__':
    torch.randn()
