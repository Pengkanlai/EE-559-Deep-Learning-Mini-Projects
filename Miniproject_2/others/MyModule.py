class MyModule:
    """
    Our base class works just like PyTorch,
    where the forward is automatically called when parameter is passed into the class object
    """
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def param(self):
        return []

    def forward(self, *args, **kwargs):
        raise NotImplementedError("You must overwrite base class forward function")

    def backward(self, *args, **kwargs):
        raise NotImplementedError("You must overwrite base class backward function")