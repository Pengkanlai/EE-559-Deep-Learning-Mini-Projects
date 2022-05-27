import os
import math
import numpy as np
from torch import optim
import torch.utils.data as data_util
from argparse import ArgumentParser
from pathlib import Path
from torchvision import transforms

try:
    from .others.Loss import *
    from .others.UNet import *
    from .others.Transform import *
except:
    from others.Loss import *
    from others.UNet import *
    from others.Transform import *


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

        ###############################################################
        ###### Modify [] inside parse_args to use in Jupyter Lab ######
        ###############################################################
        self.param = self._parse().parse_args([])

        torch.manual_seed(self.param.seed)

        self.num_epoch = 0
        self.batch_size = 4
        self.split = .2
        self.use_pretrained_model = False
        self.model_path = Path(__file__).parent / "bestmodel.pth"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lr = self.param.lr
        self.model = UNet(in_c=3)

        self.model.to(self.device)
        self.model = nn.DataParallel(self.model)

        if self.param.criterion == 'l0':
            self.criterion = L0Loss()
        elif self.param.criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif self.param.criterion == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError
        self.criterion.to(self.device)

        self.optim = optim.Adam(self.model.parameters(), lr=self.lr,
                                betas=(self.param.beta1, self.param.beta2),
                                eps=self.param.eps)

    def load_pretrained_model(self) -> None:
        self.use_pretrained_model = True
        best_model = torch.load(self.model_path, map_location=self.device)
        self.best_psnr = best_model["psnr"].detach().item()
        self.criterion = best_model["criterion"]
        self.model.load_state_dict(best_model['model_state_dict'], strict=False)

    def train(self, train_input, train_target, num_epochs) -> None:
        self.num_epoch = num_epochs
        # val_data = torch.load('val_data.pkl')

        # Initiate data loaders
        train_loader = self._dataloader(train_input, train_target)
        # val_loader = self._dataloader(*val_data)

        if os.path.exists(self.model_path):
            # Save model ONLY When PSNR gets better
            self.best_psnr = torch.load(self.model_path,
                                        map_location=self.device)['psnr'].item()
        # else:
        #     self.best_psnr = self._compute_psnr(val_loader.dataset[:][0],
        #                                         val_loader.dataset[:][1])

        for e in range(num_epochs):
            print("\n======================================")

            acc_loss = 0

            for noise, target in train_loader:
                noise = noise.to(self.device)
                target = target.to(self.device)

                self.optim.zero_grad()
                output = self.model(noise)
                loss = self.criterion(output, target)
                acc_loss = acc_loss + loss.item()

                loss.backward()
                self.optim.step()

            print(f'Epoch {e} - Avg. Training Loss per Sample {acc_loss / train_input.size(0):.4f}')


            ########################################
            ######### VALIDATION ###################
            ########################################

            # outputs = []

            # with torch.no_grad():
            #     self.model.eval()
            #     for noise, clean in val_loader:
            #         output = self.model(noise)
            #         loss = self.criterion(output, clean)
            #         acc_loss = acc_loss + loss.item()
            #
            #         outputs.append(output)
            #
            #     outputs = torch.cat(outputs, dim=0)
            #     psnr = self._compute_psnr(outputs, val_loader.dataset[:][1])
            #
            #     print(f'[BEFORE PSNR {self.best_psnr:.2f}dB]')
            #     print(f'[AFTER PSNR {psnr:.2f}dB]')
            #
            #     self._rampdown(self.optim, e, self.param.rampdown_size, verbose=True)
            #
            # if psnr > self.best_psnr:
            #     self.best_psnr = psnr
            #     print(f'[BEST PSNR {self.best_psnr:.2f}dB]')
            #
            #     torch.save({
            #         "psnr": self.best_psnr,
            #         "model_state_dict": self.model.state_dict(),
            #         "criterion": self.criterion
            #     }, self.model_path)

    @torch.no_grad()
    def predict(self, test_input) -> torch.Tensor:
        self.model.eval()
        test_input = test_input.float().to(self.device)
        output = self.model(test_input).clamp(min=0., max=255.)
        return output

    def _rampdown(self, optimizer, curr_epoch, rampdown_start_at, verbose) -> None:
        p = 1.
        rampdown_length = np.floor(rampdown_start_at * self.num_epoch)
        if curr_epoch >= (self.num_epoch - rampdown_length):
            ep = (curr_epoch - (self.num_epoch - rampdown_length)) * 0.5
            p = math.exp(-(ep * ep) / rampdown_length)

        self.lr = p * self.lr

        for param_group in optimizer.param_groups:
            if param_group['lr'] != self.lr and verbose:
                print(f"Rampdown - new lr: {self.lr}")
            param_group['lr'] = self.lr * p

    def _dataloader(self, features, label):
        """Load data into batches"""
        features = features.float().to(self.device)
        label = label.float().to(self.device)

        zipped = data_util.TensorDataset(features, label)
        dataloader = data_util.DataLoader(zipped, batch_size=self.batch_size,
                                          shuffle=False, sampler=None)

        return dataloader

    def _compute_psnr(self, denoised, clean) -> torch.Tensor:
        """Helper function to facilitate our testing and evaluation"""
        _denoised = denoised.float().div(255.0)
        _clean = clean.float().div(255.0)
        mse = torch.mean((_denoised - _clean) ** 2)
        return -10 * torch.log10(mse + 10 ** -8)

    def _parse(self):
        parser = ArgumentParser(description='Mini Project1 - PyTorch implementation of Noise2Noise')

        parser.add_argument('--seed', type=int, default=42, help='random seed')

        parser.add_argument('--lr', type=float, default=0.001,
                            help='Adam: lr')

        parser.add_argument('--beta1', type=float, default=0.9,
                            help='Adam: beta1')

        parser.add_argument('--beta2', type=float, default=0.99,
                            help='Adam: beta2')

        parser.add_argument('--eps', type=float, default=1e-8,
                            help='Adam: eps')

        parser.add_argument('--criterion', type=str, default='l2',
                            choices=['l0', 'l1', 'l2'], help='loss function')

        parser.add_argument('--noise-type', type=str, default='gaussian',
                            help='noise type')

        parser.add_argument('--rampdown-size', type=float, default=.7,
                            help='perform lr rampdown after % of epoch')

        parser.add_argument('--save-dir', type=str, default='params',
                            help='directory of saved models')

        return parser


if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys

    parser = ArgumentParser()
    parser.add_argument('-p', '--project-path', help='Path to the project folder', required=True)
    parser.add_argument('-d', '--data-path', help='Path to the data folder', required=True)

    args = parser.parse_args()

    project_path = Path(args.project_path)
    data_path = Path(args.data_path)

    sys.path.append(args.project_path)

    model = Model()
    model.load_pretrained_model()

    val_input, val_target = torch.load(data_path / 'val_data.pkl')

    out = model.predict(val_input)

    print(model._compute_psnr(out, val_target))

    # execution command: python model.py -p '/Users/xuyixuan/Downloads/Project/EE559-DL-Miniprojects/Miniproject_1' -d '/Users/xuyixuan/Downloads/Project/EE559-DL-Miniprojects'
