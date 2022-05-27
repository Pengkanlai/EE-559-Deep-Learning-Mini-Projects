import numbers
import random
import torch

import numbers


class RandomGaussianNoise(object):
    """Applying gaussian noise randomly with a given probability p"""

    def __init__(self, p=0.5, mean=0, std=0.1, fixed_distribution=True):
        assert isinstance(mean, numbers.Number) and mean >= 0, 'mean should be a positive value'
        assert isinstance(std, numbers.Number) and std >= 0, 'std should be a positive value'
        assert isinstance(p, numbers.Number) and p >= 0, 'p should be a positive value'
        self.p = p
        self.mean = mean
        self.std = std
        self.fixed_distribution = fixed_distribution

    @staticmethod
    def get_params(mean, std):
        mean = random.uniform(0, mean)
        std = random.uniform(0, std)

        return mean, std

    def gaussian_noise(self, img, mean, std):
        gauss = torch.normal(mean, std, img.shape)
        noisy = torch.clamp(gauss + img, min=0., max=255.)
        return noisy

    def __call__(self, img):
        if random.random() < self.p:
            if self.fixed_distribution:
                mean, std = self.mean, self.std
            else:
                mean, std = self.get_params(self.mean, self.std)
            return self.gaussian_noise(img, mean=mean, std=std)
        return img
