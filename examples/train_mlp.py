# examples/train_mlp.py
import numpy as np
from fudanai.core.tensor import Tensor
from fudanai.core.module import Module
from fudanai.layers.linear import Linear
from fudanai.layers.activation import ReLU
from fudanai.trainer.engine import Engine
from fudanai.optim.sgd import SGD
from fudanai.losses.cross_entropy import cross_entropy
from fudanai.data.dataset import Dataset


class MLP(Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.l1 = Linear(in_dim, hidden)
        self.l2 = Linear(hidden, out_dim)
        self._params = self.l1.parameters() + self.l2.parameters()

    def forward(self, x: Tensor):
        x = self.l1(x).relu()
        return self.l2(x)


def main():
    # 1. 造一段随机数据：500样本，20维，5类；80%训练 / 20%测试
    X = np.random.randn(500, 20)
    y = np.random.randint(0, 5, size=(500,))
    split = int(0.8 * len(X))
    train_ds = Dataset(list(X[:split]), list(y[:split]))
    test_ds = Dataset(list(X[split:]), list(y[split:]))

    # 2. 初始化模型、优化器、引擎
    model = MLP(20, 64, 5)
    optim = SGD(model.parameters(), lr=0.1)
    engine = Engine(model, optim, cross_entropy)

    # 3. 训练 & 每个 epoch 后评估一次
    epochs = 10
    for ep in range(epochs):
        print(f"===== Epoch {ep+1}/{epochs} =====")
        engine.train(train_ds, epochs=1, batch_size=32)
        engine.evaluate(test_ds, batch_size=64)

    # 4. 最终测试
    print("===== Final Evaluation =====")
    engine.evaluate(test_ds, batch_size=64)


if __name__ == '__main__':
    main()
