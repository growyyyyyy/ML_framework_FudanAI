import numpy as np
from fudanai.core.tensor import Tensor
from fudanai.core.module import Module


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = Tensor(np.random.randn(in_features, out_features)*0.01,
                        requires_grad=True)
        self.b = Tensor(np.zeros(out_features), requires_grad=True)
        self._params += [self.W, self.b]

    def forward(self, x: Tensor):
        return x @ self.W + self.b
