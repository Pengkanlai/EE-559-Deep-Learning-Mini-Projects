import torch.utils.data as data_util
import pickle
import os
from pathlib import Path

try:
    from .others.Sequential import *
    from .others.ReLU import *
    from .others.MSE import *
    from .others.SGD import *
    from .others.Sigmoid import *
    from .others.Conv2d import *
    from .others.Upsamling import *
    from .others.TransposedConv2d import *
except:
    from others.Sequential import *
    from others.ReLU import *
    from others.MSE import *
    from others.SGD import *
    from others.Sigmoid import *
    from others.Conv2d import *
    from others.Upsamling import *
    from others.TransposedConv2d import *


class Model(MyModule):
    def __init__(self, in_c=3, out_c=3):
        super(Model).__init__()

        self.model = Sequential(
            # Output: (h_in + 2 * p - k / s) + 1
            Conv2d(in_c, 24, kernel_size=4, stride=2),
            ReLU(),
            # n, 48, 15, 15

            Conv2d(24, 48, kernel_size=3, stride=2),
            ReLU(),
            # n, 48, 7, 7

            # Output: (h_in - 1) * s - 2 * p + k + out_p
            TransposedConv2d(in_channels=48, out_channels=24, padding=0,
                             kernel_size=3, stride=2, output_padding=0),
            ReLU(),
            # n, 48, 15, 15

            TransposedConv2d(in_channels=24, out_channels=out_c, padding=0,
                             kernel_size=4, stride=2, output_padding=0),
            # n, 48, 32, 32

            Sigmoid()
        )

        self.lr = 5e-1
        self.criterion = MSE()
        self.optimizer = SGD(self.model.param(), self.lr)
        self.use_pretrained_model = False
        self.best_psnr = None
        self.model_path = Path(__file__).parent / "bestmodel.pth"

    def load_pretrained_model(self) -> None:
        self.use_pretrained_model = True

        # Load best model
        with open(self.model_path, 'rb') as loader:
            best_model = pickle.load(loader)

        # Restore previous best psnr
        # self.best_psnr = best_model["best psnr"]

        # Restore the best model weight
        self.model.apply(best_model)

    def train(self, train_input, train_target, num_epochs) -> None:
        # val_input, val_target = torch.load('val_data.pkl')

        # Initiate data loaders
        train_loader = self._dataloader(train_input, train_target)

        # self.best_psnr = self._compute_psnr(val_input, val_target)

        for e in range(num_epochs):
            print("\n======================================")

            acc_loss = 0

            for noise, target in train_loader:
                self.optimizer.zero_grad()
                output = self.model(noise)
                loss = self.criterion(output, target)
                acc_loss += loss.item()

                dloss = self.criterion.backward()
                _ = self.model.backward(dloss)
                self.optimizer.step()

            print(f'Epoch {e} - Avg. Training Loss per Sample {acc_loss / train_input.size(0):.4f}')

            ########### Validation ##################
            # pred = self.model(val_input[:1000].float().div(255.))
            # psnr = self._compute_psnr(pred * 255., val_target[:1000])
            #
            # print(f'[BEST PSNR {self.best_psnr:.2f}dB]')
            # print(f'[AFTER PSNR {psnr:.2f}dB]')

            # Store best model
            # if psnr > self.best_psnr:
            #     self.best_psnr = psnr
            #     with open('bestmodel.pth', 'wb') as saver:
            #         pickle.dump(self.model.param(), saver)

    @torch.no_grad()
    def predict(self, test_input) -> torch.Tensor:
        test_input = test_input.float().div(255.)
        output = self.model(test_input)
        return output * 255.

    def _compute_psnr(self, denoised, clean) -> torch.Tensor:
        """Helper function to facilitate our testing and evaluation"""
        _denoised = denoised.float().div(255.0)
        _clean = clean.float().div(255.0)
        mse = torch.mean((_denoised - _clean) ** 2)
        return -10 * torch.log10(mse + 10 ** -8)

    def _dataloader(self, features, label):
        """Load data into batches"""
        features = features.float().div(255.)
        label = label.float().div(255.)

        zipped = data_util.TensorDataset(features, label)
        dataloader = data_util.DataLoader(zipped, batch_size=4,
                                          shuffle=False, sampler=None)

        return dataloader


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    #
    mmmmmm = Model()
    # mmmmmm.train(*torch.load('train_data.pkl'), 2)

    val_input, val_target = torch.load('val_data.pkl')

    mmmmmm.load_pretrained_model()

    out = mmmmmm.predict(val_input)
    #
    print(mmmmmm._compute_psnr(out, val_target))
    #
    # print(out)
    #
    # for n in range(595, 600):
    #     target = torch.clone(val_target[n])
    #     for i in range(3):
    #         target[i] = val_target[n][i].T
    #
    #     input = torch.clone(val_input[n])
    #     for i in range(3):
    #         input[i] = val_input[n][i].T
    #
    #     output = torch.clone(out[n])
    #     for i in range(3):
    #         output[i] = out[n][i].T
    #
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(target.T / 255)
    #     plt.title('Clean target')
    #
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(input.T / 255)
    #     plt.title('Input')
    #
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(output.T / 255)
    #     plt.title('Output')
    #
    #     plt.show()

