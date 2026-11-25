---
tags:
  - 深度学习
  - 线性回归
  - Softmax
  - PyTorch
created: 2026-08-08
---

# Linear Neural Networks

> 基于《动手学深度学习》第3章，代码使用通用 PyTorch 实现（不依赖 d2l 库）

---

## 一、线性回归（理论篇）

### 1.1 核心思想

线性回归假设输出 y 与输入特征 x 之间是**线性关系**：

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \cdots + w_d x_d + b$$

其中 **w** 是权重向量，**b** 是偏置（bias）。这本质上是一个**仿射变换**（线性变换 + 平移）。

对于整个数据集（n 个样本，d 个特征），可以用矩阵形式表示：

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b, \quad \mathbf{X} \in \mathbb{R}^{n \times d}$$

### 1.2 关键术语

| 术语 | 含义 |
|------|------|
| 样本 (sample) | 数据集中的一行数据 |
| 标签 (label) | 要预测的目标值 y |
| 特征 (feature) | 输入的自变量 x |
| 权重 (weight) | 每个特征的影响系数 w |
| 偏置 (bias) | 所有特征为 0 时的预测值 b |
| 超参数 (hyperparameter) | 手动设置的参数（如学习率、批量大小） |

### 1.3 损失函数

采用**均方误差（MSE）**作为损失函数：

$$L(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^{n} \frac{1}{2} \left( \hat{y}^{(i)} - y^{(i)} \right)^2$$

前面的 1/2 是为了求导后消去系数 2，使公式更简洁。

### 1.4 优化方法

**解析解**（仅线性回归适用）：

$$\mathbf{w}^* = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

**小批量随机梯度下降（Mini-batch SGD）**（通用方法）：

$$(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \nabla_{(\mathbf{w},b)} l^{(i)}$$

其中 η 是学习率，B 是小批量样本集合。

### 1.5 正态分布与平方损失的联系

假设观测噪声服从正态分布 ε ~ N(0, σ²)，即 y = w⊤x + b + ε，则**最小化均方误差等价于对线性模型做极大似然估计**。这为使用 MSE 提供了概率论上的理论依据。

### 1.6 线性回归即单层神经网络

线性回归可以看作只有一个输出神经元的**全连接层**（没有激活函数），输入层有 d 个节点，输出层有 1 个节点。

---

## 二、线性回归 — 从零实现

### 2.1 生成合成数据

```python
import torch
import random
import matplotlib.pyplot as plt

def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # 添加高斯噪声
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

**代码细节解读**：

`true_w = torch.tensor([2.0, -3.4])` 创建了一个包含两个元素的张量，表示**人为设定的"真实权重"**。之所以叫 `true_w`，是因为这是我们预设的标准答案，后面训练的目标就是让模型学出尽可能接近 `[2.0, -3.4]` 的权重。两个元素意味着数据有 **2 个特征**。

`X = torch.normal(0, 1, (num_examples, len(w)))` 从**标准正态分布**（均值=0，标准差=1）中随机采样，生成形状为 `(num_examples, len(w))` 的矩阵：

| 参数 | 值 | 含义 |
|------|---|------|
| `0` | 均值 μ | 正态分布的中心 |
| `1` | 标准差 σ | 正态分布的离散程度 |
| `(num_examples, len(w))` | 形状 | 矩阵的维度 |

如果 `num_examples=1000`，最终 X 的形状是 **(1000, 2)**：1000 个样本（行），每个样本有 2 个特征（列）。生成出来的 X 大概长这样：

```
tensor([[ 0.31, -1.24],    # 第1个样本的两个特征值
        [ 0.78,  0.56],    # 第2个样本
        [-1.02,  0.33],    # 第3个样本
        ...                # 共 1000 行
       ])
```

### 2.2 小批量数据迭代器

```python
def data_iter(batch_size, features, labels):
    """随机打乱后按小批量生成数据"""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_idx = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_idx], labels[batch_idx]
```

**`list(range(num_examples))` 的作用**：

`range(num_examples)` 生成一个从 0 到 num_examples-1 的整数序列，但它是一个**惰性对象**（不是真正的列表），不会一次性把所有数字放进内存。而 `random.shuffle` 要求输入是一个**可变的列表**，所以必须先用 `list()` 转换。

流程：
```python
indices = list(range(5))     # [0, 1, 2, 3, 4]
random.shuffle(indices)       # 比如变成 [3, 0, 4, 1, 2]
```

之后按打乱后的索引去取数据，就实现了**随机抽取小批量样本**的效果，确保每个 epoch 中样本的访问顺序不同，这对 SGD 的训练效果很重要。

### 2.3 初始化参数

```python
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

### 2.4 定义模型、损失函数、优化器

```python
def linreg(X, w, b):
    """线性回归前向传播"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    """均方损失（不取均值，留给优化器处理）"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

**SGD 函数详解**：

函数签名 `def sgd(params, lr, batch_size)` 的三个参数：

| 参数 | 传入的实际值 | 含义 |
|------|------------|------|
| `params` | `[w, b]` | 模型的所有参数，打包成一个**列表** |
| `lr` | 如 `0.03` | 学习率（learning rate） |
| `batch_size` | 如 `10` | 小批量的样本数 |

调用时写的是 `sgd([w, b], lr, batch_size)`，所以 `for param in params` 就是依次遍历这个列表：
- 第1次循环: `param = w`（权重张量，形状 (2, 1)）
- 第2次循环: `param = b`（偏置张量，形状 (1,)）

**`param` 就是模型中每一个需要更新的参数张量**，用同样的梯度下降公式分别更新它们。

`with torch.no_grad():` 告诉 PyTorch **下面这些操作纯粹是数值更新，不要记录梯度计算图**。因为 `param -= ...` 本身也是张量运算，如果不关闭梯度追踪，PyTorch 会把这次"更新参数"的操作也纳入计算图，导致下一次 `backward()` 时计算出错误的梯度，同时还浪费内存。

核心更新公式 `param -= lr * param.grad / batch_size` 对应的数学公式是：

$$\theta \leftarrow \theta - \frac{\eta}{|\mathcal{B}|} \cdot \nabla_\theta L$$

拆解每一部分：
- `param` → θ，当前参数的值
- `param.grad` → ∇L，PyTorch 自动求导算出的梯度
- `lr` → η，学习率
- `batch_size` → |B|，批量大小

**为什么要除以 `batch_size`？** 因为在训练循环中我们调用的是 `loss.sum().backward()`，梯度是整个小批量所有样本损失的**总和**的梯度，除以 batch_size 就变成了**平均梯度**，让更新步长不随批量大小变化。

`param.grad.zero_()` 是清零梯度。PyTorch 的梯度默认是**累加**的，即每次调用 `backward()` 新算出的梯度会加到 `param.grad` 上面，而不是替换。如果不清零，下一个 batch 的梯度就会和上一个 batch 的梯度混在一起，导致更新完全错误。末尾的下划线 `_` 是 PyTorch 的惯例，表示**原地操作**（in-place）。

### 2.5 训练循环

```python
lr = 0.03
num_epochs = 3
batch_size = 10

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        loss = squared_loss(linreg(X, w, b), y)
        loss.sum().backward()       # 反向传播，计算梯度
        sgd([w, b], lr, batch_size) # 更新参数
    with torch.no_grad():
        train_loss = squared_loss(linreg(features, w, b), labels)
        print(f'epoch {epoch+1}, loss {float(train_loss.mean()):.6f}')

print(f'w 的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b 的估计误差: {true_b - b}')
```

---

## 三、线性回归 — 简洁实现

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------- 1. 数据 ---------
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
X = torch.normal(0, 1, (1000, 2))
y = torch.matmul(X, true_w) + true_b + torch.normal(0, 0.01, (1000,))
y = y.reshape((-1, 1))

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# --------- 2. 模型 ---------
model = nn.Linear(2, 1)              # 输入特征数=2, 输出=1
model.weight.data.normal_(0, 0.01)   # 初始化权重
model.bias.data.fill_(0)             # 初始化偏置

# --------- 3. 损失和优化器 ---------
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

# --------- 4. 训练 ---------
for epoch in range(3):
    for X_batch, y_batch in dataloader:
        loss = criterion(model(X_batch), y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 3.1 TensorDataset 的作用

`TensorDataset(X, y)` 把特征张量 X 和标签张量 y **打包成一个数据集对象**，使得可以通过索引同时取出对应的一对 (特征, 标签)。

打包前，X 和 y 是两个独立的张量，需要手动保证索引对齐。打包后，一次索引就能取出配对好的数据：

```python
dataset[0]  # → (tensor([1., 2.]), tensor([7.]))
dataset[1]  # → (tensor([3., 4.]), tensor([8.]))
len(dataset) # → 3
```

之所以需要它，是因为下一步要交给 `DataLoader` 使用，而 `DataLoader` 要求输入是一个**数据集对象**（支持 `__len__` 和 `__getitem__`），而不是两个零散的张量。整个流程是：

```
X, y (两个独立张量)
    ↓  TensorDataset 打包
dataset (一个数据集对象，支持按索引取 (x, y) 对)
    ↓  DataLoader 封装
dataloader (自动 shuffle、分 batch、多线程加载)
```

如果没有 `TensorDataset` 把 X 和 y 绑定在一起，`DataLoader` 就无法知道哪个 x 对应哪个 y。

### 3.2 nn.Linear 内部的权重形状

**通用规则**：`nn.Linear(m, n)` 内部自动创建：

| 参数 | 形状 | 含义 |
|------|------|------|
| `weight` | **(n, m)** | 权重矩阵 |
| `bias` | **(n,)** | 偏置向量 |

注意是 **(n, m)** 而不是 (m, n)，容易搞反。例子：

```python
nn.Linear(2, 1)     → weight shape: (1, 2), bias shape: (1,)
nn.Linear(784, 10)  → weight shape: (10, 784), bias shape: (10,)
nn.Linear(3, 5)     → weight shape: (5, 3), bias shape: (5,)
```

之所以是 (n, m)，是因为 `nn.Linear` 内部的计算公式是 $\text{output} = x \cdot W^\top + b$。输入 x 形状为 `(batch_size, m)`，所以需要 W 形状为 `(n, m)`，转置后 `(m, n)` 才能做矩阵乘法：

```
  x        ×     Wᵀ       +   b      =   output
(batch, m) × (m, n)       + (n,)     = (batch, n)
```

只需要记住 **`nn.Linear(输入维度, 输出维度)`**，权重形状就是**反过来的 `(输出, 输入)`**。

以 `nn.Linear(2, 1)` 为例，虽然只有 1 个输出，但内部的 weight 是一个 `(1, 2)` 的矩阵，**里面有 2 个权重**对应 w₁ 和 w₂，而不是只有 1 个。`model.weight.data.normal_(0, 0.01)` 是**对整个张量操作**，把 weight 张量中的每一个元素都用 N(0, 0.01) 的随机值填充。

### 3.3 从零实现 vs 简洁实现对比

| 组件 | 从零实现 | 简洁实现 |
|------|---------|---------|
| 数据加载 | 手写 `data_iter` 生成器 | `DataLoader` + `TensorDataset` |
| 模型 | 手写 `linreg` 函数 | `nn.Linear(2, 1)` |
| 损失函数 | 手写 `squared_loss` | `nn.MSELoss()` |
| 优化器 | 手写 `sgd` 函数 | `torch.optim.SGD()` |
| 梯度管理 | 手动 `param.grad.zero_()` | `optimizer.zero_grad()` |

---

## 四、Softmax 回归（理论篇）

### 4.1 从回归到分类

线性回归解决的是**回归问题**（预测连续值），而 softmax 回归解决的是**分类问题**（预测离散类别）。

对于 q 个类别的分类：

$$o_j = \sum_{i=1}^{d} w_{ji} x_i + b_j, \quad j = 1, 2, \ldots, q$$

矩阵形式：**o = Wx + b**，其中 W ∈ ℝ^{q×d}，b ∈ ℝ^q。

**注意区分 x 和 y**：x 是输入特征，**不是**独热编码。独热编码是**标签 y** 的表示方式。

以 Fashion-MNIST 为例，每张图片是 28×28 = 784 个像素，展平后 x 就是一个 784 维的向量，每个元素是一个像素的灰度值（0~1之间的连续数）：

```
x = [0.0, 0.0, 0.12, 0.85, 0.93, 0.71, ..., 0.0, 0.0]
     ←—————————————— 784 个浮点数 ——————————————→
```

这和独热编码完全不同，x 里的值可以是任意浮点数。

**公式在干什么**：假设只有 3 个特征（d=3），3 个类别（q=3），x = [0.5, 0.8, 0.2]。公式展开后其实就是**每个类别各算一个分数**：

```
o₁ = w₁₁·x₁ + w₁₂·x₂ + w₁₃·x₃ + b₁   ← 属于第1类的分数
o₂ = w₂₁·x₁ + w₂₂·x₂ + w₂₃·x₃ + b₂   ← 属于第2类的分数
o₃ = w₃₁·x₁ + w₃₂·x₂ + w₃₃·x₃ + b₃   ← 属于第3类的分数
```

代入具体数值：

```
W = [[0.1, 0.2, 0.3],    b = [0.1,
     [0.4, 0.5, 0.6],         0.2,
     [0.7, 0.8, 0.9]]         0.3]

o₁ = 0.1×0.5 + 0.2×0.8 + 0.3×0.2 + 0.1 = 0.37
o₂ = 0.4×0.5 + 0.5×0.8 + 0.6×0.2 + 0.2 = 0.92
o₃ = 0.7×0.5 + 0.8×0.8 + 0.9×0.2 + 0.3 = 1.47
```

所以 o = [0.37, 0.92, 1.47]。这三个数叫做 **logits**（原始分数），数值越大表示模型越倾向于认为输入属于该类别。此时 o₃ 最大，模型倾向预测为第 3 类。

### 4.2 独热编码（One-Hot Encoding）

将类别标签编码为向量。例如 3 个类别中的第 2 类：y = (0, 1, 0)。

### 4.3 Softmax 函数

o 有两个问题：可能是负数，而且不求和为 1，不是合法的概率。所以需要 **softmax** 把它变成概率分布：

$$\hat{y}_j = \text{softmax}(o_j) = \frac{\exp(o_j)}{\sum_{k=1}^{q} \exp(o_k)}$$

性质：所有 ŷ_j ≥ 0 且 Σ ŷ_j = 1。

沿用上例：

```
ŷ₁ = exp(0.37) / (exp(0.37) + exp(0.92) + exp(1.47)) = 0.185
ŷ₂ = exp(0.92) / (...)                                 = 0.321
ŷ₃ = exp(1.47) / (...)                                 = 0.556
```

现在可以说：模型预测该样本有 18.5% 概率属于第1类、32.1% 属于第2类、55.6% 属于第3类。

### 4.4 交叉熵损失与为何不用 MSE

分类任务使用**交叉熵损失**：

$$l(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{j=1}^{q} y_j \log \hat{y}_j$$

由于 y 是独热编码，只有真实类别对应的那一项非零，因此简化为 $l = -\log \hat{y}_{[\text{真实类别}]}$。

**为什么分类不用 MSE？三个角度**：

**① 梯度角度（最实际）**：假设真实标签是第 1 类，用 MSE 时：

$$L = (\hat{y}_1 - 1)^2, \quad \frac{\partial L}{\partial o} \propto (\hat{y}_1 - 1) \cdot \hat{y}_1(1-\hat{y}_1)$$

当模型预测非常错误时，比如 ŷ₁ = 0.01，此时 ŷ₁(1-ŷ₁) ≈ 0.01，梯度极小，**梯度消失**。而交叉熵：

$$L = -\log \hat{y}_1, \quad \frac{\partial L}{\partial o} \propto \hat{y}_1 - 1$$

当 ŷ₁ = 0.01 时，梯度 = -0.99，非常大。错得越离谱，梯度越大。

```
预测概率 ŷ₁    MSE 梯度大小    交叉熵梯度大小
──────────    ───────────    ─────────────
0.01（大错）    ≈ 0.01 😴      0.99 🔥
0.30（偏错）    ≈ 0.15         0.70
0.90（接近）    ≈ 0.01         0.10
0.99（很准）    ≈ 0.0001       0.01
```

**② 概率论角度**：

```
回归问题 → 输出是连续值 → 假设高斯噪声 → 推出 MSE
分类问题 → 输出是概率   → 假设多项分布 → 推出 交叉熵
```

**③ 直觉角度**：假设真实标签 [1, 0, 0]，两个模型：

```
模型C: [0.99, 0.005, 0.005]  → 非常确信是猫
模型D: [0.01, 0.495, 0.495]  → 几乎确信不是猫
```

MSE 下，模型D 损失 ≈ 1.47；交叉熵下，模型D 损失 = -log(0.01) = 4.61，惩罚极重。交叉熵在模型"自信地犯错"时给出**极大的惩罚**，这正是分类任务需要的特性。

### 4.5 Softmax 导数推导

为什么 MSE 的梯度里会出现 ŷ₁(1-ŷ₁) 这一项？关键是 softmax 对 logit 的导数。

**o 是什么**：o 是 softmax **之前**的原始分数（logit），就是线性变换的直接输出：

```
x → 线性变换 o = Wx + b → o (原始分数) → softmax → ŷ → 损失 L
```

用**链式法则**：

$$\frac{\partial L}{\partial o} = \frac{\partial L}{\partial \hat{y}_1} \cdot \frac{\partial \hat{y}_1}{\partial o}$$

**Softmax 的导数**（对自身类别 o₁）：

$$\hat{y}_1 = \frac{e^{o_1}}{\sum_k e^{o_k}} = \frac{e^{o_1}}{S}$$

设分母为 S，用商法则。**关键：S 对 o₁ 求导为什么是 e^{o₁}？**

S 是所有类别的指数之和：$S = e^{o_1} + e^{o_2} + \cdots + e^{o_q}$。对 o₁ 求导时，只有 $e^{o_1}$ 那一项含 o₁，其他项对 o₁ 都是常数，导数为 0：

```
∂/∂o₁ (e^{o₁})  =  e^{o₁}    ← 只有这一项含 o₁
∂/∂o₁ (e^{o₂})  =  0         ← o₂ 和 o₁ 无关
    ...
```

所以 $\frac{\partial S}{\partial o_1} = e^{o_1}$。

商法则 $(f/g)' = (f'g - fg')/g^2$，其中 f = $e^{o_1}$，g = S：

$$\frac{\partial \hat{y}_1}{\partial o_1} = \frac{e^{o_1} \cdot S - e^{o_1} \cdot e^{o_1}}{S^2} = \frac{e^{o_1}}{S} \cdot \frac{S - e^{o_1}}{S} = \hat{y}_1(1 - \hat{y}_1)$$

这个形式和 sigmoid 的导数一样。**问题就出在 ŷ₁(1-ŷ₁) 这一项**，它是 softmax 导数自带的，当预测概率接近 0 或 1 时都趋近于 0，把梯度"压没了"。而交叉熵的导数 ∂L/∂o = ŷ₁ - 1，**没有这个乘项**，所以不会消失。

---

## 五、图像分类数据集（Fashion-MNIST）

### 5.1 数据集简介

Fashion-MNIST 包含 10 类服饰图片，每张图片为 28×28 的灰度图。训练集 60,000 张，测试集 10,000 张。

类别：T恤、裤子、套衫、连衣裙、外套、凉鞋、衬衫、运动鞋、包、短靴。

### 5.2 加载数据集

```python
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()  # 将 PIL 图像转为 [0,1] 范围的张量

train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, transform=transform, download=True)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=4)
```

### 5.3 transform 是怎么起作用的

`transform = transforms.ToTensor()` 这行只是创建了一个"转换器"对象，还没有处理任何图片。`ToTensor()` 做三件事：

| 操作 | 之前 | 之后 |
|------|------|------|
| 数据类型 | PIL 图像或 numpy 数组 | PyTorch 张量 (Tensor) |
| 数值范围 | 像素值 0~255 的整数 | 0.0~1.0 的浮点数 |
| 维度顺序 | (高, 宽, 通道) | (通道, 高, 宽) |

以 Fashion-MNIST 为例，一张灰度图从 `(28, 28)` 变成 `(1, 28, 28)` 的张量。

传给数据集时（`transform=transform`），数据集会**每次取一张图片时自动调用它**。伪代码：

```python
def __getitem__(self, i):
    image = 读取第i张原始PIL图片()    # 像素值 0~255
    label = 读取第i个标签()

    if self.transform is not None:
        image = self.transform(image)  # ← 在这里自动调用 ToTensor()

    return image, label
```

### 5.4 张量 vs 数组

张量本质上就是**多维数组**，但比 numpy 数组多了两个关键能力：

**① 自动求导**（最核心的区别）：

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x
y.backward()
print(x.grad)  # → tensor([7.])  即 2x+3 在 x=2 时的值
```

**② 可以在 GPU 上运算**：

```python
t_gpu = t.to('cuda')  # 速度快几十到几百倍
```

总结：
```
numpy 数组   =  多维数组（纯数据容器）
PyTorch 张量 =  多维数组 + 自动求导 + GPU加速
```

日常理解时把它当数组想完全没问题，只是它是**专门为深度学习增强过的数组**。

### 5.5 num_workers 参数

`num_workers` 是数据加载的**子进程数量**，决定用多少个额外的进程来并行读取数据、做预处理。

```
num_workers=0（默认）
  主进程自己去读数据，读完再训练
  → 做菜和拿食材交替进行，GPU 常空闲

num_workers=4
  4个子进程同时读数据，提前备好下几个 batch
  → GPU 永远不用等，一直在训练
```

一般设为 CPU 核心数的一半或 2~4 就够了。Windows 上有时会报错，可以先设成 0。

---

## 六、Softmax 回归 — 从零实现

```python
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# =================== 数据加载 ===================
transform = transforms.ToTensor()
train_data = torchvision.datasets.FashionMNIST(
    './data', train=True, transform=transform, download=True)
test_data = torchvision.datasets.FashionMNIST(
    './data', train=False, transform=transform, download=True)

batch_size = 256
train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_data, batch_size, shuffle=False, num_workers=4)

# =================== 初始化参数 ===================
num_inputs = 784   # 28 * 28
num_outputs = 10   # 10 个类别

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# =================== 核心函数 ===================
def softmax(O):
    """数值稳定的 softmax"""
    O_exp = torch.exp(O - O.max(dim=1, keepdim=True).values)
    return O_exp / O_exp.sum(dim=1, keepdim=True)

def net(X):
    """模型前向传播：将图片展平后做线性变换 + softmax"""
    return softmax(torch.matmul(X.reshape((-1, num_inputs)), W) + b)

def cross_entropy(y_hat, y):
    """交叉熵损失"""
    return -torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    preds = y_hat.argmax(dim=1)
    return (preds == y).float().sum().item()

def evaluate_accuracy(data_loader, net):
    """在数据集上评估模型精度"""
    correct, total = 0.0, 0
    with torch.no_grad():
        for X, y in data_loader:
            y_hat = net(X)
            correct += accuracy(y_hat, y)
            total += y.numel()
    return correct / total

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# =================== 训练 ===================
lr = 0.1
num_epochs = 10

for epoch in range(num_epochs):
    total_loss, total_correct, total_samples = 0.0, 0.0, 0
    for X, y in train_loader:
        y_hat = net(X)
        loss = cross_entropy(y_hat, y)
        loss.sum().backward()
        sgd([W, b], lr, batch_size)
        total_loss += loss.sum().item()
        total_correct += accuracy(y_hat, y)
        total_samples += y.numel()
    test_acc = evaluate_accuracy(test_loader, net)
    print(f'epoch {epoch+1}, '
          f'loss {total_loss/total_samples:.4f}, '
          f'train acc {total_correct/total_samples:.3f}, '
          f'test acc {test_acc:.3f}')
```

### 6.1 argmax(dim=1) 详解

`argmax(dim=1)` 的含义是**沿着第1维（列方向），找出最大值所在的索引**。

**关键理解：dim 指的是"沿着哪个维度操作"，也就是哪个维度会被消掉**。

```
          col0  col1  col2
          ↓     ↓     ↓
row0 →  [0.1,  0.7,  0.2]
row1 →  [0.8,  0.1,  0.1]

形状: (2, 3)
       ↑  ↑
     dim0 dim1
```

**`dim=1`** 表示沿着 dim=1（列方向）扫过去，把这个维度压缩掉：

```
row0:  0.1 → 0.7 → 0.2  扫一遍，找到最大值在下标 1
row1:  0.8 → 0.1 → 0.1  扫一遍，找到最大值在下标 0

结果: [1, 0]   形状从 (2, 3) 变成了 (2,)  ← dim=1 那个维度 3 被消掉了
```

**记忆口诀**：`dim = 哪个维度被消掉`

| 操作 | 消掉的维度 | 效果 |
|------|----------|------|
| `argmax(dim=0)` | 行被消掉 | 每一列里找最大 |
| `argmax(dim=1)` | 列被消掉 | 每一行里找最大 |

分类任务中永远用 `dim=1`，因为每一**行**是一个样本的各类别概率，我们要的是每个样本概率最大的那个类别。

### 6.2 准确率计算链式操作

`return (preds == y).float().sum().item()` 是在**计算这个 batch 预测正确的数量**。

假设 `preds = tensor([1, 0, 2, 1])`, `y = tensor([1, 0, 2, 0])`：

```
preds == y                → tensor([True, True, True, False])   # 逐元素比较
    ↓ .float()
                          → tensor([1., 1., 1., 0.])           # 布尔转浮点
    ↓ .sum()
                          → tensor(3.)                          # 求和（还是张量）
    ↓ .item()
                          → 3.0                                 # 转为普通 Python 数字
```

**为什么每一步都需要**：
- `.float()`：布尔值不能直接参与数值计算，转成浮点后 True→1.0，False→0.0
- `.sum()`：把所有 1 加起来，得到预测正确的数量
- `.item()`：从 0 维张量中取出普通 Python 数字，方便后续用普通加减法累加，且打印更干净

### 6.3 numel() 元素个数

`numel()` **不是求和**，是计算张量中**元素的总个数**。名字来源是 **num**ber of **el**ements 的缩写。

```python
a = torch.tensor([1, 2, 3])
a.numel()  # → 3

b = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])
b.numel()  # → 6（2×3）

c = torch.zeros(2, 3, 4)
c.numel()  # → 24
```

在训练代码中，`y.numel()` 就是这个 batch 的样本数。

区分：

| 函数 | 作用 | 例子 |
|------|------|------|
| `numel()` | 元素**个数** | `tensor([1,2,3]).numel()` → `3` |
| `sum()` | 元素**求和** | `tensor([1,2,3]).sum()` → `tensor(6)` |
| `len()` | 第0维的**长度** | `len(tensor([[1,2],[3,4]]))` → `2` |

### 6.4 关键要点

**softmax 数值稳定性**：直接计算 exp(o) 可能溢出。技巧是先减去每行的最大值：exp(o - max(o))，结果不变但避免了数值问题。

**交叉熵损失的巧妙实现**：利用 `y_hat[range(n), y]` 直接索引出每个样本真实类别的预测概率，无需构造独热编码。

---

## 七、Softmax 回归 — 简洁实现

```python
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# =================== 数据 ===================
transform = transforms.ToTensor()
train_data = torchvision.datasets.FashionMNIST(
    './data', train=True, transform=transform, download=True)
test_data = torchvision.datasets.FashionMNIST(
    './data', train=False, transform=transform, download=True)
batch_size = 256
train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_data, batch_size, shuffle=False, num_workers=4)

# =================== 模型 ===================
model = nn.Sequential(
    nn.Flatten(),        # 将 (batch, 1, 28, 28) 展平为 (batch, 784)
    nn.Linear(784, 10)   # 全连接层：784 -> 10
)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)

model.apply(init_weights)

# =================== 损失和优化器 ===================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# =================== 评估函数 ===================
def evaluate_accuracy(loader, model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    model.train()
    return correct / total

# =================== 训练 ===================
num_epochs = 10

for epoch in range(num_epochs):
    total_loss, correct, total = 0.0, 0, 0
    for X, y in train_loader:
        y_hat = model(X)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.numel()
        correct += (y_hat.argmax(dim=1) == y).sum().item()
        total += y.numel()
    test_acc = evaluate_accuracy(test_loader, model)
    print(f'epoch {epoch+1}, '
          f'loss {total_loss/total:.4f}, '
          f'train acc {correct/total:.3f}, '
          f'test acc {test_acc:.3f}')
```

### 7.1 nn.CrossEntropyLoss 的关键

PyTorch 的 `CrossEntropyLoss` 将 **softmax + 负对数似然** 合并为一步计算。因此模型的最后一层**不需要**手动加 softmax，直接输出原始分数（logits）即可。这样做的好处是数值更稳定（利用了 log-sum-exp 技巧）。

### 7.2 init_weights 与 model.apply

**定义初始化规则**：

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
```

这个函数接收一个**层对象 m**，检查是不是 `Linear`（全连接层），如果是就用正态分布 N(0, 0.01) 重新填充权重。

**`isinstance()` 是 Python 内置函数**，用来判断一个对象是不是某个类型：

```python
isinstance(对象, 类型)   # 返回 True 或 False

isinstance(10, int)                    # → True
isinstance(nn.Flatten(), nn.Linear)    # → False
isinstance(nn.Linear(2,1), nn.Linear)  # → True
```

它和 `type()` 的区别：`isinstance` 会沿着继承链往上查，认可继承关系；`type()` 只看精确类型。实际开发中几乎总是用 `isinstance`。

**应用到整个模型**：`model.apply(init_weights)` 会**自动遍历模型中的每一个层**，把每个层依次传给 `init_weights` 函数：

```
model.apply(init_weights)
  │
  ├─ 传入 Flatten 层     → isinstance(Flatten, nn.Linear) → False → 跳过
  │
  ├─ 传入 Linear(784, 10) → isinstance(Linear, nn.Linear) → True
  │   └─ nn.init.normal_(weight, std=0.01)
  │      权重全部替换为 N(0, 0.01) 的随机值
  │
  └─ 传入 Sequential 本身 → 不是 Linear → 跳过
```

对多层网络特别方便，一行代码就能初始化所有 Linear 层。

### 7.3 total_loss += loss.item() * y.numel() 详解

`CrossEntropyLoss` 默认 `reduction='mean'`，返回的是这个 batch 所有样本损失的**平均值**，不是总和。

拆解 `total_loss += loss.item() * y.numel()`：

| 部分 | 含义 | 例子 |
|------|------|------|
| `loss.item()` | 取出平均损失的 Python 数值 | `0.35` |
| `y.numel()` | 这个 batch 的样本数 | `256` |
| 两者相乘 | 平均损失 × 样本数 = **这个 batch 的总损失** | `89.6` |
| `+=` | 累加到 `total_loss` 中 | 跨所有 batch 累计 |

本质就是**把平均值还原成总和**。

**为什么不直接用 `loss.item()` 累加？** 因为**最后一个 batch 可能不满**。假设总共 1000 个样本，batch_size=256，最后一个 batch 只有 232 个样本：

```
错误做法：每个 batch 权重相同
avg = (0.35 + 0.30 + 0.28 + 0.25) / 4 = 0.295
# batch 4 只有 232 个样本，却和 256 个样本的 batch 同等对待，不公平

正确做法：先还原总损失再除以总样本数
total_loss = 0.35×256 + 0.30×256 + 0.28×256 + 0.25×232 = 296.08
total_samples = 256 + 256 + 256 + 232 = 1000
avg_loss = 296.08 / 1000 = 0.29608   ← 准确的全局平均损失
```

### 7.4 评估函数中如何使用训练后的参数

评估函数中 `model(X)` 这一句直接使用了当前模型的参数计算 y_hat。这里并没有显式地加载什么 checkpoint 文件，因为 **model 对象本身就携带着训练好的参数**。

**参数存在哪里**：

```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10)   # 内部有 weight 和 bias
)
```

训练循环中的 `optimizer.step()` 会**直接原地修改** `model` 内部的 `weight` 和 `bias`，不是创建一个新模型：

```
① model = nn.Sequential(...)              # 参数是随机初始化的
          ↓
② 训练循环 optimizer.step()               # 参数被反复原地更新
          ↓
③ evaluate_accuracy(test_loader, model)   # 传入的 model 就是训练后的
          ↓
④ model(X)                                # 用训练后的参数做前向传播
```

第③步传入的 `model` 和第①步创建的是**同一个 Python 对象**，只是内部参数经过训练被改变了。Python 中对象是引用传递，所以函数里的 `model` 和外面的 `model` 指向同一块内存。

只有当把模型保存后、在另一个脚本中重新加载时，才需要显式的 `load_state_dict`。在同一个脚本中训练完紧接着评估，不需要这一步。

---

## 八、全章知识脉络总结

```
线性神经网络
├── 回归任务（线性回归）
│   ├── 模型：ŷ = Xw + b
│   ├── 损失：均方误差 MSE
│   ├── 优化：解析解 / SGD
│   └── 本质：单个输出神经元的全连接层
│
├── 分类任务（Softmax 回归）
│   ├── 模型：o = Xw + b → ŷ = softmax(o)
│   ├── 损失：交叉熵
│   ├── 优化：SGD（无解析解）
│   └── 本质：多个输出神经元的全连接层 + softmax
│
└── 通用训练流程
    ① 准备数据 → DataLoader (shuffle + batch)
    ② 定义模型 → nn.Linear / nn.Sequential
    ③ 选择损失 → MSELoss / CrossEntropyLoss
    ④ 选择优化器 → SGD / Adam
    ⑤ 训练循环：
       for epoch:
         for batch:
           forward → loss → backward → update
    ⑥ 评估 → 在测试集上计算 loss / accuracy
```

### 核心对比表

| 维度 | 线性回归 | Softmax 回归 |
|------|---------|-------------|
| 任务类型 | 回归（连续值） | 分类（离散类别） |
| 输出数量 | 1 | q（类别数） |
| 输出变换 | 无 | softmax |
| 损失函数 | 均方误差 | 交叉熵 |
| 评估指标 | MSE / R² | 准确率 (accuracy) |
| 解析解 | 存在 | 不存在 |
| 标签编码 | 数值 | 独热编码（one-hot） |
| PyTorch 损失 | `nn.MSELoss()` | `nn.CrossEntropyLoss()` |

### 需要记住的数学直觉

1. **MSE 等价于高斯噪声下的极大似然估计** — 为什么回归用 MSE
2. **交叉熵度量两个概率分布的差异** — 为什么分类用交叉熵
3. **Softmax 保证输出是概率分布** — 非负且求和为 1
4. **Softmax 导数带 ŷ(1-ŷ) 项** — 这是 MSE + softmax 梯度消失的根源
5. **SGD 中学习率太大会振荡，太小会收敛慢** — 需要调参
6. **批量大小影响梯度估计的方差** — 小批量噪声大但泛化可能更好

### PyTorch 使用要点回顾

- **张量 = 数组 + 自动求导 + GPU 加速**
- **`nn.Linear(m, n)` 内部 weight 形状是 `(n, m)`**，不是 (m, n)
- **`dim` 参数指"哪个维度被消掉"**，argmax(dim=1) 是每行找最大
- **`numel()` 是元素个数，不是求和**
- **`isinstance()` 优于 `type()`**，认可继承关系
- **`model` 对象通过引用传递**，训练修改的是同一对象的内部参数
- **平均损失累加要乘 batch 大小再除以总样本数**，处理不满 batch
- **`nn.CrossEntropyLoss` 自带 softmax**，模型最后一层输出 logits 即可
- **`torch.no_grad()` 用于评估和参数更新**，避免不必要的计算图
