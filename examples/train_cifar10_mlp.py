
import os
import urllib.request
import tarfile
import pickle
import numpy as np

from fudanai.core.tensor import Tensor
from fudanai.core.module import Module
from fudanai.layers.linear import Linear
from fudanai.layers.activation import ReLU
from fudanai.trainer.engine import Engine
from fudanai.optim.sgd import SGD
from fudanai.losses.cross_entropy import cross_entropy
from fudanai.data.dataset import Dataset

CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR_DIR = './data/cifar-10-batches-py'


def download_and_extract():
    if not os.path.isdir(CIFAR_DIR):
        os.makedirs('data', exist_ok=True)
        archive = 'data/cifar-10-python.tar.gz'
        print(f'Downloading CIFAR-10 to {archive}...')
        urllib.request.urlretrieve(CIFAR_URL, archive)
        print('Extracting...')
        with tarfile.open(archive, 'r:gz') as tf:
            tf.extractall(path='data')
        print('Done.')


def load_batch(batch_file):
    with open(batch_file, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    data = d[b'data']      # shape (10000, 3072)
    labels = d[b'labels']  # list of length 10000
    # 归一化到 [0,1]
    return data.astype(np.float32) / 255.0, np.array(labels, dtype=np.int64)


def load_cifar10():
    download_and_extract()
    # 训练集合并 5 个 batch
    train_X, train_y = [], []
    for i in range(1, 6):
        xf, yf = load_batch(os.path.join(CIFAR_DIR, f'data_batch_{i}'))
        train_X.append(xf)
        train_y.append(yf)
    X_train = np.concatenate(train_X, axis=0)
    y_train = np.concatenate(train_y, axis=0)
    # 测试集
    X_test, y_test = load_batch(os.path.join(CIFAR_DIR, 'test_batch'))
    return (X_train, y_train), (X_test, y_test)


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
    # 加载数据
    (X_train, y_train), (X_test, y_test) = load_cifar10()

    # 为了演示速度，随机抽取子集
    N_train, N_test = 5000, 1000
    train_ds = Dataset(
        [x for x in X_train[:N_train]],
        [int(y) for y in y_train[:N_train]]
    )
    test_ds = Dataset(
        [x for x in X_test[:N_test]],
        [int(y) for y in y_test[:N_test]]
    )

    # 初始化模型、优化器、引擎
    model = MLP(32*32*3, 100, 10)
    optim = SGD(model.parameters(), lr=0.1)
    engine = Engine(model, optim, cross_entropy)

    # 训练 & 每个 epoch 后评估一次
    epochs = 5
    for ep in range(epochs):
        print(f"===== Epoch {ep+1}/{epochs} =====")
        engine.train(train_ds, epochs=1, batch_size=64)
        engine.evaluate(test_ds, batch_size=128)

    # 最终测试
    print("===== Final Evaluation =====")
    engine.evaluate(test_ds, batch_size=128)


if __name__ == '__main__':
    main()
