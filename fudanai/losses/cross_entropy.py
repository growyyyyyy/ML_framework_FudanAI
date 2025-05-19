import numpy as np
from fudanai.core.tensor import Tensor


def cross_entropy(logits: Tensor, labels):
    # logits.data: (N, C)
    exps = np.exp(logits.data - np.max(logits.data, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    N = logits.data.shape[0]
    loss = -np.log(probs[np.arange(N), labels]).mean()
    out = Tensor(loss, requires_grad=True)

    # 简单构建 backward (不完全支持向量化)
    class CEContext:
        def __init__(self, logits, labels, out):
            self.parents = (logits,)
            self.labels = labels
            self.output = out

        def backward(self, grad_output):
            probs_copy = probs.copy()
            probs_copy[np.arange(N), self.labels] -= 1
            grads = probs_copy / N * grad_output
            return (Tensor(grads),)

    out._ctx = CEContext(logits, labels, out)
    return out
