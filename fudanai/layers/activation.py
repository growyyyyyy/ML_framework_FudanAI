from fudanai.core.tensor import Tensor
from fudanai.core.module import Module


class ReLU(Module):
    def forward(self, x: Tensor):
        return x.relu()
