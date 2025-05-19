FudanAI/
├── fudanai/
│   ├── core/
│   │   ├── tensor.py
│   │   └── module.py
│   ├── layers/
│   │   ├── linear.py
│   │   └── activation.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── dataloader.py
│   ├── losses/
│   │   └── cross_entropy.py
│   ├── optim/
│   │   └── sgd.py
│   ├── trainer/
│   │   └── engine.py
│   └── __init__.py
├── examples/
│   └── train_mlp.py
├── setup.py
└── README.md

# FudanAI

纯自研深度学习小框架，支持：
- Tensor + 自动求导
- Module、Linear、ReLU 等层
- DataLoader、Dataset
- CrossEntropy、SGD
- 简易 Engine 训练循环

## 快速开始

```bash
uv sync
python examples/train_mlp.py
```