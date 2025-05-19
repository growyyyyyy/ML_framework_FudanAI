class Module:
    def __init__(self):
        self._params = []

    def parameters(self):
        return self._params

    def zero_grad(self):
        for p in self._params:
            p.grad = None

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)
