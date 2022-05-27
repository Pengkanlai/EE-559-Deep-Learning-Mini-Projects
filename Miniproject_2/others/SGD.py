import torch


class SGD:

    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for layer in self.params:
            for wb, grad in layer:
                if grad is not None:
                    grad.zero_()

    def step(self):
        for layer in self.params:
            for wb, grad in layer:
                if wb is not None and (grad is not None):
                    with torch.no_grad():
                        wb.add_(-self.lr * grad)
