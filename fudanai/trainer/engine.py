# fudanai/trainer/engine.py
import numpy as np
from fudanai.data.dataloader import DataLoader
from fudanai.core.tensor import Tensor


class Engine:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optim = optimizer
        self.loss_fn = loss_fn

    def train(self, dataset, epochs=1, batch_size=32):
        loader = DataLoader(dataset, batch_size, shuffle=True)
        for ep in range(epochs):
            total_loss = 0
            count = 0
            for xs, ys in loader:
                # 构造批次 Tensor
                batch_x = Tensor(np.stack(xs), requires_grad=False)
                logits = self.model(batch_x)
                loss = self.loss_fn(logits, ys)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += loss.data
                count += 1
            print(
                f"[Train] Epoch {ep+1}/{epochs}, loss={total_loss/count:.4f}")

    def evaluate(self, dataset, batch_size=32):
        """
        在给定数据集上做一次前向推理，计算 avg loss 和 accuracy
        """
        loader = DataLoader(dataset, batch_size, shuffle=False)
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for xs, ys in loader:
            # 前向不需梯度
            batch_x = Tensor(np.stack(xs), requires_grad=False)
            logits = self.model(batch_x)
            loss = self.loss_fn(logits, ys)
            total_loss += loss.data

            # 计算预测
            preds = np.argmax(logits.data, axis=1)
            total_correct += (preds == np.array(ys)).sum()
            total_samples += len(ys)

        avg_loss = total_loss / len(loader)
        accuracy = total_correct / total_samples * 100
        print(f"[Eval] loss={avg_loss:.4f}, accuracy={accuracy:.2f}%")
        return avg_loss, accuracy
