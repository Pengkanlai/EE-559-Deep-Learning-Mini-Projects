from .MyModule import *


class MSE(MyModule):

    def __init__(self) -> None:
        super().__init__()
        self.pred = None
        self.target = None
        self.output_gradient = None
        self.loss = None

    def forward(self, pred, target) -> None:
        self.pred = pred
        self.target = target
        self.loss = (pred - target).abs().pow(2).mean()
        return self.loss

    def backward(self):
        self.output_gradient = 2 * (self.pred - self.target)/(self.pred.size(0)*self.pred.size(1)*self.pred.size(2)*self.pred.size(3))
        return self.output_gradient

    def param(self):
        return [(self.loss, self.output_gradient)]
