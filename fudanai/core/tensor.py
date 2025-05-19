# fudanai/core/tensor.py

import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False):
        """
        data: 可以是标量、list、ndarray，最终存为 ndarray
        requires_grad: 是否记录梯度
        """
        self.data = np.array(data, dtype=float)
        self.grad = None
        self.requires_grad = requires_grad
        self._ctx = None  # 运算上下文

    def backward(self, grad: np.ndarray = None):
        """
        向后传播梯度。所有 ctx.backward 返回的必须是 ndarray。
        这里统一把 raw ndarray 包成 Tensor 并累加到 parent.grad。
        """
        if not self.requires_grad:
            return

        # 1) 根节点 grad：ndarray
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = Tensor(grad)

        # 2) 收集所有计算上下文
        ctxs, visited = [], set()

        def build(v):
            if v._ctx and v not in visited:
                visited.add(v)
                for p in v._ctx.parents:
                    build(p)
                ctxs.append(v._ctx)
        build(self)

        # 3) 反向传播
        for ctx in reversed(ctxs):
            # 拿到 output.grad，转换成 ndarray
            out_grad = ctx.output.grad.data if isinstance(ctx.output.grad, Tensor) \
                else ctx.output.grad

            # 调用纯 numpy 的 backward，必须返回 ndarray 或 tuple of ndarray
            raw_grads = ctx.backward(out_grad)
            if not isinstance(raw_grads, (list, tuple)):
                raw_grads = (raw_grads,)

            # 对每个 parent，包装并累加
            for parent, raw in zip(ctx.parents, raw_grads):
                arr = raw.data if isinstance(raw, Tensor) else raw
                grad_tensor = Tensor(arr, requires_grad=parent.requires_grad)

                if parent.grad is None:
                    parent.grad = grad_tensor
                else:
                    summed = parent.grad.data + arr
                    parent.grad = Tensor(
                        summed, requires_grad=parent.requires_grad)

    # —— 基本算子 —— #

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad)

        class AddCtx:
            def __init__(self, a, b, out):
                self.parents = (a, b)
                self.output = out

            def backward(self, grad_output: np.ndarray):
                # 返回两个 ndarray，不要包装 Tensor
                # grad w.r.t. xW 部分，形状不变
                grad_a = grad_output
                # grad w.r.t. b 部分，要沿 batch 维度求和
                grad_b = grad_output.sum(axis=0)
                return grad_a, grad_b

        out._ctx = AddCtx(self, other, out)
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad)

        class MulCtx:
            def __init__(self, a, b, out):
                self.parents = (a, b)
                self.output = out

            def backward(self, grad_output: np.ndarray):
                a, b = self.parents
                grad_a = grad_output * b.data
                grad_b = grad_output * a.data
                return grad_a, grad_b

        out._ctx = MulCtx(self, other, out)
        return out

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad)

        class MatMulCtx:
            def __init__(self, a, b, out):
                self.parents = (a, b)
                self.output = out

            def backward(self, grad_output: np.ndarray):
                a, b = self.parents
                grad_a = grad_output @ b.data.T
                grad_b = a.data.T @ grad_output
                return grad_a, grad_b

        out._ctx = MatMulCtx(self, other, out)
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data),
                     requires_grad=self.requires_grad)

        class ReluCtx:
            def __init__(self, a, out):
                self.parents = (a,)
                self.output = out

            def backward(self, grad_output: np.ndarray):
                a = self.parents[0]
                grad = grad_output.copy()
                grad[a.data < 0] = 0
                return grad

        out._ctx = ReluCtx(self, out)
        return out

    def numpy(self):
        """返回底层 ndarray"""
        return self.data
