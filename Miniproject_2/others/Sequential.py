from .MyModule import *


class Sequential(MyModule):
    def __init__(self, *layers) -> None:
        super().__init__()
        self.modules = []
        for layer in layers:
            self.modules.append(layer)

    def forward(self, x):
        res = x
        for layer in self.modules:
            res = layer.forward(res)
        return res

    def backward(self, grad):
        grad_from_back = grad
        for layer in reversed(self.modules):
            grad_from_back = layer.backward(grad_from_back)
        return grad_from_back

    def param(self):
        res = []
        for layer in self.modules:
            res.append(layer.param())
        return res

    def apply(self, parameters):
        for idx, wb in enumerate(parameters):
            if wb[0][0] is not None and wb[1][0] is not None:
                self.modules[idx].weight = parameters[idx][0][0]
                self.modules[idx].bias = parameters[idx][1][0]


if __name__ == '__main__':
    import torch
    from torch import nn
    from Conv2d import *
    from TransposedConv2d import *
    from ReLU import *
    from SGD import *
    from Sigmoid import *
    from MSE import *

    # PyTorch Sequential
    pppp = nn.Sequential(

        # Output: (h_in + 2 * p - k / s) + 1
        nn.Conv2d(3, 48, kernel_size=2, stride=2),
        nn.ReLU(),
        # n, 48, 16, 16

        nn.Conv2d(48, 48, kernel_size=2, stride=2),
        nn.ReLU(),
        # n, 48, 8, 8

        # Output: (h_in - 1) * s - 2 * p + k + out_p
        nn.ConvTranspose2d(in_channels=48, out_channels=48, padding=1,
                           kernel_size=3, stride=2, output_padding=1),
        nn.ReLU(),
        # n, 48, 16, 16

        nn.ConvTranspose2d(in_channels=48, out_channels=3, padding=1,
                           kernel_size=3, stride=2, output_padding=1),
        # n, 48, 32, 32

        nn.Sigmoid()
    )

    c1 = Conv2d(in_channels=3, out_channels=48, kernel_size=2, stride=2)
    c1.weight = pppp[0].weight
    c1.bias = pppp[0].bias

    c2 = Conv2d(in_channels=48, out_channels=48, kernel_size=2, stride=2)
    c2.weight = pppp[2].weight
    c2.bias = pppp[2].bias

    tc1 = TransposedConv2d(in_channels=48, out_channels=48, padding=1, kernel_size=3, stride=2, output_padding=1)
    tc1.weight = pppp[4].weight
    tc1.bias = pppp[4].bias

    tc2 = TransposedConv2d(in_channels=48, out_channels=3, padding=1, kernel_size=3, stride=2, output_padding=1)
    tc2.weight = pppp[6].weight
    tc2.bias = pppp[6].bias

    ctc = Sequential(c1,
                     ReLU(),
                     c2,
                     ReLU(),
                     tc1,
                     ReLU(),
                     tc2,
                     Sigmoid())

    optimizer = SGD(ctc.param(), 0.001)
    PYTopt = torch.optim.SGD(pppp.parameters(), 0.001)

    n1, n2 = torch.load('../train_data.pkl')

    for j in range(3):
        for i in range(0, 100, 4):
            # print(j*50000 + i)

            xxxxxx = n1[i:i + 4].float().div(255.)
            yyyyyy = n2[i:i + 4].float().div(255.)
            xxxxxx.requires_grad_()

            optimizer.zero_grad()
            PYTopt.zero_grad()

            out = pppp(xxxxxx)
            loss = nn.MSELoss()
            l = loss(out, yyyyyy)
            l.backward()

            pred = ctc(xxxxxx)
            ll = MSE()
            gd = ll(pred, yyyyyy)
            test_grad = ll.backward()
            x_grad = ctc.backward(test_grad)

            optimizer.step()
            PYTopt.step()

            # pred
            torch.testing.assert_allclose(pppp(xxxxxx), ctc(xxxxxx))

            # dl_dx
            torch.testing.assert_allclose(xxxxxx.grad, x_grad)

            # dl_dw
            torch.testing.assert_allclose(pppp[0].weight.grad, ctc.param()[0][0][1])
            torch.testing.assert_allclose(pppp[2].weight.grad, ctc.param()[2][0][1])
            torch.testing.assert_allclose(pppp[4].weight.grad, ctc.param()[4][0][1])
            torch.testing.assert_allclose(pppp[6].weight.grad, ctc.param()[6][0][1])

            # dl_db
            torch.testing.assert_allclose(pppp[0].bias.grad, ctc.param()[0][1][1])
            torch.testing.assert_allclose(pppp[2].bias.grad, ctc.param()[2][1][1])
            torch.testing.assert_allclose(pppp[4].bias.grad, ctc.param()[4][1][1])
            torch.testing.assert_allclose(pppp[6].bias.grad, ctc.param()[6][1][1])
