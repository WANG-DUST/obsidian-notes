
---

## 目录

- [第一部分：CNN 基础](#第一部分cnn-基础)
  - 1. 从 MLP 到卷积
  - 2. 图像卷积（互相关运算）
  - 3. 填充与步幅
  - 4. 多输入多输出通道
  - 5. 汇聚层（Pooling）
  - 6. LeNet
- [第二部分：现代 CNN](#第二部分现代-cnn)
  - 7. AlexNet
  - 8. VGG
  - 9. NiN
  - 10. GoogLeNet
  - 11. 批量规范化（Batch Normalization）
  - 12. **ResNet（重点）**
  - 13. DenseNet
- [第三部分：附录](#第三部分附录)
  - A. 参数量与显存估算
  - B. 演进主线与横向对比
  - C. 通用 CNN 设计原则

---

# 第一部分：CNN 基础

## 1. 从 MLP 到卷积

### 1.1 动机：MLP 处理图像的困境

假设一张 1000×1000 的 RGB 图像被展平送入 MLP，若第一隐藏层有 1000 个神经元，参数量就是 3×10⁹，训练不动、存不下、还会严重过拟合。更重要的是，展平丢掉了「相邻像素强相关」这一图像最重要的先验。

### 1.2 两条先验：不变性与局部性

CNN 的整个设计可以由两条对图像的先验推出：

1. **平移不变性（translation invariance / equivariance）**：识别一个物体不应依赖它在图像中的位置。检测器参数不该随位置变化。
2. **局部性（locality）**：早期的图像特征（边缘、纹理）只依赖于像素的局部邻域。

### 1.3 从数学上推导卷积

对全连接层，将输入和权重都索引成二维：

```
h[i,j] = Σ_{k,l} w[i,j,k,l] · x[k,l]
       = Σ_{a,b} v[i,j,a,b] · x[i+a, j+b]
```

- **平移不变性** 要求 `v[i,j,a,b]` 不依赖 `(i,j)`，即 `v[i,j,a,b] = v[a,b]`：

  ```
  h[i,j] = Σ_{a,b} v[a,b] · x[i+a, j+b]
  ```

- **局部性** 要求 `|a|, |b| > Δ` 时 `v[a,b] = 0`：

  ```
  h[i,j] = Σ_{a=-Δ..Δ} Σ_{b=-Δ..Δ} v[a,b] · x[i+a, j+b]
  ```

这就是**卷积层**。参数从 O(HW · HW) 骤降到 O((2Δ+1)²)。

---

## 2. 图像卷积（互相关运算）

### 2.1 互相关 vs 卷积

深度学习框架实现的其实是**互相关（cross-correlation）**，不是数学上严格意义的卷积。区别只在于卷积核是否需要翻转 180°。因为卷积核是学出来的，翻不翻转在训练后是等价的，所以大家都直接叫「卷积」。

对二维输入 `X ∈ R^(n_h × n_w)` 和核 `K ∈ R^(k_h × k_w)`：

```
Y[i,j] = Σ_{a=0..k_h-1} Σ_{b=0..k_w-1} X[i+a, j+b] · K[a,b]
```

输出形状为 `(n_h - k_h + 1) × (n_w - k_w + 1)`。

### 2.2 从零实现互相关（用于理解，不用于生产）

```python
import torch

def corr2d(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """二维互相关运算"""
    kh, kw = K.shape
    out_h, out_w = X.shape[0] - kh + 1, X.shape[1] - kw + 1
    Y = torch.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            Y[i, j] = (X[i:i+kh, j:j+kw] * K).sum()
    return Y
```

### 2.3 边缘检测的例子

用一个 `[[1, -1]]` 的核可以检测「颜色变化」的**竖直边缘**：

```python
X = torch.ones((6, 8))
X[:, 2:6] = 0                    # 中间是黑色矩形
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
# Y 的第 2 列是 1（黑→白），第 6 列是 -1（白→黑），其余为 0
```

### 2.4 学习卷积核

给定输入和期望输出，可以像训练全连接层一样，用梯度下降学出卷积核：

```python
import torch.nn as nn

conv2d = nn.Conv2d(in_channels=1, out_channels=1,
                   kernel_size=(1, 2), bias=False)

X_ = X.reshape(1, 1, 6, 8)
Y_ = Y.reshape(1, 1, 6, 7)

for i in range(10):
    Y_hat = conv2d(X_)
    loss = ((Y_hat - Y_) ** 2).sum()
    conv2d.zero_grad()
    loss.backward()
    with torch.no_grad():
        conv2d.weight -= 3e-2 * conv2d.weight.grad

print(conv2d.weight.data.reshape(1, 2))   # 会逼近 [1, -1]
```

### 2.5 特征映射与感受野（Receptive Field）

- **特征映射（feature map）**：卷积层的输出，可看作在空间维度上被学出来的一组特征。
- **感受野（receptive field）**：某个输出位置能「看到」的输入区域大小。
  - 一层 3×3 卷积：感受野 3×3
  - 两层 3×3 卷积堆叠：感受野扩大到 5×5
  - 三层堆叠：7×7

**这是 VGG 用小卷积核堆叠代替大核的理论基础**：两层 3×3 和一层 5×5 感受野相同，但参数更少、非线性更多。

---

## 3. 填充与步幅

设输入形状 `n_h × n_w`，核 `k_h × k_w`，四周填充 `p_h, p_w`（上下/左右总量），步幅 `s_h, s_w`。输出形状：

```
⌊(n_h - k_h + p_h + s_h) / s_h⌋ × ⌊(n_w - k_w + p_w + s_w) / s_w⌋
```

常见约定：
- **保持尺寸不变**（same padding）：核大小为奇数 `k`，两边各填 `⌊k/2⌋`。
- **下采样为原来的 1/2**：核为奇数，步幅 2，same padding。

```python
# 保持形状不变：3x3 卷积 + padding=1
conv_same = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)

# 下采样一半：3x3 卷积 + stride=2 + padding=1
conv_down = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2)
```

---

## 4. 多输入多输出通道

### 4.1 多输入通道

输入 `X ∈ R^(c_i × h × w)`，核 `K ∈ R^(c_i × k_h × k_w)`。**每个通道分别做二维互相关，然后按通道求和**：

```
Y[i,j] = Σ_{c=0..c_i-1} Σ_{a,b} X[c, i+a, j+b] · K[c, a, b]
```

从零实现（复用 §2.2 的 `corr2d`），把每个通道的互相关结果加起来：

```python
import torch

def corr2d_multi_in(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """多输入通道的互相关：X 形状 (c_i, h, w)，K 形状 (c_i, kh, kw)。"""
    # 对 X 和 K 的第 0 维（通道）逐个做 corr2d 再求和
    return sum(corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],       # 通道 0
                  [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])      # 通道 1
K = torch.tensor([[[0., 1.], [2., 3.]],
                  [[1., 2.], [3., 4.]]])
print(corr2d_multi_in(X, K))
# tensor([[ 56.,  72.],
#         [104., 120.]])
```

用 `nn.Conv2d` 直接做（工程上永远这样写）：

```python
import torch.nn as nn

# 2 通道输入 → 1 通道输出，2x2 卷积
conv = nn.Conv2d(in_channels=2, out_channels=1,
                 kernel_size=2, bias=False)

# 手动把权重设成上面的 K，验证等价
with torch.no_grad():
    conv.weight.copy_(K.reshape(1, 2, 2, 2))  # (c_out, c_in, kh, kw)

X_ = X.reshape(1, 2, 3, 3)                    # (N, C, H, W)
print(conv(X_).squeeze())                     # 与 corr2d_multi_in 结果相同
```

### 4.2 多输出通道

要输出 `c_o` 个通道，就需要 `c_o` 组核，每组独立地作用在所有输入通道上：

```
K ∈ R^(c_o × c_i × k_h × k_w)
```

在 PyTorch 中，`nn.Conv2d(c_in, c_out, kernel_size=k)` 的权重形状就是 `(c_out, c_in, k, k)`。

从零实现（在 §4.1 基础上，对 K 的第 0 维——输出通道——再遍历一次）：

```python
def corr2d_multi_in_out(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """多输入多输出通道的互相关：
    X 形状 (c_i, h, w)，K 形状 (c_o, c_i, kh, kw)，返回 (c_o, out_h, out_w)。"""
    # 对 K 的每一个「输出通道核组」k 分别做多输入互相关，再沿通道维堆叠
    return torch.stack([corr2d_multi_in(X, k) for k in K], dim=0)

# 构造 3 组核（3 个输出通道）：K、K+1、K+2
K_multi = torch.stack([K, K + 1, K + 2], dim=0)
print(K_multi.shape)                # torch.Size([3, 2, 2, 2])  = (c_o, c_i, kh, kw)
print(corr2d_multi_in_out(X, K_multi).shape)   # torch.Size([3, 2, 2])
```

用 `nn.Conv2d` 的实际写法：

```python
# 3 通道输入 → 64 通道输出，3x3 卷积
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)

print(conv.weight.shape)   # torch.Size([64, 3, 3, 3])  = (c_out, c_in, kh, kw)
print(conv.bias.shape)     # torch.Size([64])           = (c_out,)

X = torch.randn(1, 3, 32, 32)      # (N, C, H, W)
print(conv(X).shape)               # torch.Size([1, 64, 32, 32])
```

**记忆**：`nn.Conv2d` 的权重是**四维张量** `(c_out, c_in, kh, kw)`，参数量 = `c_out × c_in × kh × kw`（加上 `c_out` 个 bias）。这个公式在 §A.1 的参数量估算中反复用到。

### 4.3 1×1 卷积：跨通道的信息融合

1×1 卷积失去了「识别空间模式」的能力，但保留了「按通道混合」的能力。它等价于对每个像素做一次通道维度上的全连接。

**用途**：
- **通道数升降维**（比 3×3 卷积便宜得多）
- **跨通道信息融合**
- 在 NiN、GoogLeNet、ResNet 的 bottleneck 中都是关键组件

```python
# 1x1 卷积：把 64 通道降到 16 通道，空间尺寸不变
conv1x1 = nn.Conv2d(64, 16, kernel_size=1)
```

### 4.4 通道数为什么随深度递增

看后面几个网络的架构会发现一个规律：随着网络加深、空间尺寸下采样，**通道数总是成倍增加**。VGG 是 64 → 128 → 256 → 512，AlexNet 是 96 → 256 → 384 → 256，ResNet 也是 64 → 128 → 256 → 512。这不是随意选的，背后有三条相互印证的理由。

**(1) 信息量的「补偿」与容量守恒**

下采样（池化或大步幅卷积）虽然去除了冗余，但也不可避免地压缩了空间信息（比如物体的精确物理坐标、极其微小的纹理细节）。为了保证网络在层层传递中不会因为图像变小而丢失总的特征信息量，网络必须在**深度维度**（也就是通道数）上进行补偿。

可以把每一层的输出特征图想象成一个长方体水容器：下采样把水容器的「底面积」（空间 H × W）缩小到四分之一，为了装下提纯后同样多甚至更复杂的「特征之水」，水容器的「高度」（通道数）就必须相应增加。

**(2) 匹配特征语义的复杂度**

每一个「通道」都可以视作一个特定的特征检测器。而网络的浅层和深层负责提取完全不同级别的特征：

- **浅层**（通道数少，如 64）：提取简单、底层的特征，比如横向边缘、纵向边缘、特定的颜色斑块。这些基础构成元素的种类相对有限。
- **深层**（通道数多，如 256、512）：把底层的边缘和斑块组合成高级的、具有具体语义的特征，比如「猫的耳朵」「汽车的轮胎」「人的眼睛」等。

高级语义概念的种类数远远多于底层的简单几何线条，所以网络越深，就需要成倍增加的通道数（更多的检测器）来捕捉、分类并存储这些海量的高级抽象概念。

**(3) 保持层与层之间的计算负荷平衡**

这一条体现了架构设计的工程智慧。一个卷积层的计算量（FLOPs）大致与「特征图面积 × 输入通道数 × 输出通道数」成正比。以 ResNet 为例，每次遇到 `stride=2` 的下采样时：

- 特征图长宽各减半，**整体面积变成原来的 1/4**
- 同时把输入和输出通道数**各翻一倍**（即 `2 × 2 = 4` 倍）

这样一来，面积的缩小（1/4）和通道乘积的增加（4×）刚好抵消。整个网络无论在浅层还是深层，**每一层的计算压力都大致保持恒定**，避免了头重脚轻或计算资源分配不均导致显存溢出的问题。

**验算**：设某 stage 内部的典型块（stage 内已完成通道翻倍后）为 `Conv(128, 128, 3x3)`，特征图 `(H/2, W/2)`；上一 stage 的典型块为 `Conv(64, 64, 3x3)`，特征图 `(H, W)`。两者 FLOPs 之比：

```
[(H/2)(W/2) · 128 · 128 · 9]  /  [H · W · 64 · 64 · 9]
= (1/4) · (128/64)²
= (1/4) · 4
= 1
```

每 stage 的每块计算量精确相等。参数量随通道数平方增长的规律参见 §A.2。

---

## 5. 汇聚层（Pooling）

汇聚层没有可学习参数，作用是：
1. **降低空间分辨率**（降低后续计算量、增加感受野）
2. **提升平移不变性**（对输入的小平移不那么敏感）
3. **聚合信息**（把邻域内的最强/平均响应挑出来）

两种常见汇聚：
- **最大汇聚（Max Pooling）**：取窗口最大值，突出最显著特征
- **平均汇聚（Average Pooling）**：取窗口均值，更平滑；**全局平均池化（Global Average Pooling）**常用在网络末端代替全连接层

**关键点**：汇聚在每个通道上独立进行，不改变通道数。

```python
nn.MaxPool2d(kernel_size=2, stride=2)      # 2x2 最大池化
nn.AvgPool2d(kernel_size=2, stride=2)      # 2x2 平均池化
nn.AdaptiveAvgPool2d((1, 1))               # 全局平均池化
```

---

## 6. LeNet：第一个成功的 CNN

Yann LeCun 1998 年提出，用于手写数字识别。是「卷积 + 池化 → 全连接」范式的起源。

```python
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),          nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84),         nn.Sigmoid(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x): return self.classifier(self.features(x))
```

LeNet 的历史意义是「证明了在 GPU 出现之前 CNN 也能工作」。真正把 CNN 推到主流的是十几年后的 AlexNet。

---

# 第二部分：现代 CNN

从这里开始，网络越来越深、模块化设计越来越强，训练技巧（BN、残差连接、精心初始化等）也越来越重要。

## 7. AlexNet

2012 年 ImageNet 冠军，深度学习复兴的标志。相对 LeNet 的关键改进：

| 变化 | 意义 |
|---|---|
| 网络加深（8 层）、加宽 | 更强的表达能力 |
| **ReLU** 替代 Sigmoid | 缓解梯度消失，训练加速 |
| **Dropout** | 全连接层正则化 |
| **数据增强**（裁剪、翻转、颜色抖动） | 缓解过拟合 |
| **最大池化**替代平均池化 | 更突出显著特征 |
| GPU 训练 | 让上面这一切成为可能 |

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 4096),        nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x): return self.classifier(self.features(x))
```

---

## 8. VGG：使用块的网络

Simonyan & Zisserman, 2014。核心贡献是**「块」（block）的抽象**：不再手工设计每一层，而是重复堆叠一个标准化的 VGG 块。

**VGG 块**：若干个「3×3 conv + ReLU」堆叠，末尾接一个 2×2 max pool（步幅 2）。

**关键观察**：**多个小卷积堆叠 > 一个大卷积**
- 两层 3×3 与一层 5×5 感受野相同，但参数少（2·9c² vs 25c²），非线性多
- 三层 3×3 与一层 7×7 感受野相同

```python
def make_vgg_block(num_convs: int, in_channels: int, out_channels: int) -> nn.Sequential:
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

class VGG(nn.Module):
    """conv_arch 是 [(num_convs, out_channels), ...]"""
    def __init__(self, conv_arch, in_channels: int = 3, num_classes: int = 1000):
        super().__init__()
        blocks = []
        for num_convs, out_c in conv_arch:
            blocks.append(make_vgg_block(num_convs, in_channels, out_c))
            in_channels = out_c
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_c * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 4096),          nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x): return self.classifier(self.features(x))

# VGG-11
vgg11_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
```

VGG 的思想（用块拼网络）直接启发了后续所有现代 CNN 的设计。

---

## 9. NiN：网络中的网络

Lin et al., 2013。两个关键想法：

1. **NiN 块** = `Conv(k×k) → ReLU → Conv(1×1) → ReLU → Conv(1×1) → ReLU`
   - 用两层 1×1 卷积在每个像素位置上做「微型 MLP」，增强通道间的非线性组合
2. **用全局平均池化（Global Average Pooling）替代全连接层**
   - 网络末端不再是「Flatten → Linear」，而是把最后一个卷积层的输出通道数直接设为类别数，然后对每个通道做全局平均池化
   - **好处**：参数量大减、不易过拟合、更容易解释（每个通道对应一个类别的响应图）

**这个「用 1×1 conv + GAP 代替 FC」的思路被 GoogLeNet、ResNet 全盘继承。**

```python
def nin_block(in_c, out_c, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size, stride, padding), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=1), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=1), nn.ReLU(inplace=True),
    )
```

---

## 10. GoogLeNet：含并行连结的网络

Szegedy et al., 2014。核心创新是 **Inception 块**：**同一层内用不同尺寸的卷积核并行提取特征，然后按通道拼接**。

**Inception 块的四条路径**：
1. `1×1 conv`
2. `1×1 conv → 3×3 conv`
3. `1×1 conv → 5×5 conv`
4. `3×3 max pool → 1×1 conv`

四条路径的输出在通道维度上拼接（`torch.cat(dim=1)`）。

**为什么每条路径前都有 1×1 conv？** 用来先把通道数降下来，控制计算量。这是 1×1 卷积作为「瓶颈」（bottleneck）的第一个成熟应用。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception(nn.Module):
    def __init__(self, in_c, c1, c2, c3, c4):
        super().__init__()
        self.p1 = nn.Conv2d(in_c, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)
```

---

## 11. 批量规范化（Batch Normalization）

Ioffe & Szegedy, 2015。**它不是一种网络架构，而是几乎所有现代网络都要用的一种层。**

### 11.1 想解决的问题：内部协变量偏移与训练不稳定

深度网络训练难，一个直觉性的解释是**内部协变量偏移（internal covariate shift, ICS）**：底层参数一变，后面所有层的输入分布就跟着变。这个「分布漂移」和**学习率的选择**深度绑定，构成了训练时的一个 tradeoff。

**大学习率的场景**：底层参数一次更新的幅度是 `lr · ∇L`。如果 `lr` 大，第一层输出的分布就会有一次大幅漂移（比如均值从 0 跳到 5，方差扩大 10 倍）。这个新分布进入第二层时，第二层的权重是**针对旧分布调好的**——比如它可能已经把一半神经元推到了 sigmoid 的饱和区，或者输出恰好在 ReLU 的死区边缘。分布一变，激活值可能爆炸（10× 的方差经过几层就是 10⁶ 倍）或者塌陷到 0。前向一炸，反向梯度跟着炸，参数更新变成 NaN——**发散**。层数越深，这个连锁反应越严重。

**小学习率的场景**：每一步走得小，分布漂移也小，训练是稳的——但代价是走得慢，需要非常多的 epoch 才能收敛。而且小 `lr` 不能解决另一个问题：即使收敛过程稳定，激活值也可能在某个不好的分布上「卡住」（比如全都进了饱和区），此时梯度接近 0，学得再久也不动。

**BN 打破这个 tradeoff**：BN 在每一层入口把分布强行拉回 `(0, 1)`（然后用可学的 `γ, β` 让网络自己决定要不要偏离）。这样底层参数怎么变，第二层看到的输入分布**基本恒定**——第二层的权重不再需要追着一个漂移的目标跑。于是大 `lr` 不会把后续层的输入推到未知区域，训练不再发散；用 SGD + `lr=0.1` 训练 100+ 层，这在 BN 之前是做不到的。ResNet 论文里实测加了 BN 后学习率可以调大一个数量级。

**一个重要的补充**：ICS 可能不是真正原因。2018 年 MIT 的一篇论文 *How Does Batch Normalization Help Optimization?*（Santurkar et al., NeurIPS 2018）做了个有趣的实验：他们在 BN 后**人为注入噪声**，让每层的输入分布重新变得剧烈漂移（ICS 甚至比不加 BN 还严重），结果发现——**训练依然又快又稳**。他们的结论是：BN 真正的作用是**让损失曲面变得更平滑**（loss landscape 的 Lipschitz 常数和 Hessian 谱都变好），使得梯度方向更可靠、允许更大步长。ICS 的减少只是副产品，不是根本原因。所以更准确的说法是：「训不动 → 需要小 lr」这个现象是真的，BN 允许大 lr 也是真的；但中间的机制到底是 ICS 还是曲面平滑，学界更倾向后者。

### 11.2 具体做法：训练阶段（核心 4 步）

BN 的核心思想是：在训练过程中，强行将每一层输入的数据分布拉回到均值为 0、方差为 1 的标准正态分布，然后再进行可学习的缩放和平移。这样做完美解决了「激活值掉入饱和区导致梯度消失」的问题。

假设训练时送入网络的一个批次（Mini-Batch）有 m 个样本。对于某一层网络的一个特定神经元（或特征图的一个特定通道），它的输入集合为 `B = {x₁, x₂, ..., x_m}`。BN 的操作严格按照以下 4 步进行：

**第 1 步：计算当前批次的均值（Mean）**

求这 m 个数据的平均值：

```
μ_B = (1/m) Σ_{i=1..m} x_i
```

**第 2 步：计算当前批次的方差（Variance）**

求这 m 个数据偏离均值的程度：

```
σ_B² = (1/m) Σ_{i=1..m} (x_i - μ_B)²
```

**第 3 步：归一化（Normalize）**

将数据减去均值，并除以标准差，使其变成均值为 0、方差为 1 的分布：

```
x_hat_i = (x_i - μ_B) / sqrt(σ_B² + ε)
```

分母里的 ε（epsilon）是一个极小的常数（如 10⁻⁵），仅仅是为了防止分母为 0 导致除零异常。

**第 4 步：缩放与平移（Scale and Shift）——最精妙的一步**

```
y_i = γ · x_hat_i + β
```

这一步是 γ（gamma，缩放因子）和 β（beta，平移因子）的由来。如果只做前 3 步，数据全被强行按在 0 附近——虽然解决了饱和问题，但也抹杀了网络原本学到的特征表达能力（比如有些特征本来就应该分布在边缘）。因此 BN 引入了这两个**可学习参数**，它们跟着网络一起通过梯度下降学出来。网络会自己决定：「我是应该保持标准正态分布，还是稍微拉宽一点、偏移一点？」

**极端情况下**：如果网络认为没必要做归一化，它可以让 `γ = sqrt(σ_B² + ε)` 并且 `β = μ_B`，从而完美还原出原始的 `x_i`——这保证了 BN 层不会削弱网络的表达能力。

**这里很容易产生一个疑问**：既然归一化把数据拉到了均值 0、方差 1，然后又用 γ 和 β 把它缩放和平移走，这不就等于把前面做的事给撤销了吗？BN 到底还有什么意义？答案的核心在于：BN 赋予了网络「撤销归一化」的**能力**，但网络**绝不会**把它变回原本那个失控的、糟糕的分布。具体从三个维度来理解：

**(a) 改变了控制权：从「蝴蝶效应」到「直接掌控」**

在没有 BN 的时候，第 10 层神经网络的输入分布，是由前面 9 层所有的权重矩阵（数以万计的参数）相乘相加决定的。这就像「蝴蝶效应」：第 1 层的权重稍微变动一点点，经过 9 层的放大，第 10 层的分布就会发生剧烈的偏移——分布很容易失控地跑到均值为 100 或者 -500 的地方，直接掉进饱和区，梯度瞄间死亡。

加了 BN 和 γ、β 之后，情况完全不同了。第一步的归一化相当于「强行踩下刹车」，把前面 9 层累积的混乱分布直接清零（均值归 0，方差归 1），切断了蝴蝶效应。第二步的 γ 和 β 是**唯二**决定这一层最终分布的参数。现在第 10 层的分布不再受前面 9 层无数参数的牵制，而是仅仅由 γ 和 β 两个参数直接且稳定地控制。网络只需要轻轻调整这两个参数，就能让分布处于最完美的位置。这种「解耦」让优化变得极其容易，Loss 曲面会非常平滑（这也在 §11.1 提到的 Santurkar 论文里得到了实验验证）。

**(b) 为什么要脱离 (0, 1) 分布：拯救非线性**

如果只做归一化，把数据死死地钉在均值 0、方差 1 的位置，会带来一个灾难性的后果：**丧失非线性表达能力**。

拿 Sigmoid 来看：在 0 附近的中心区域，Sigmoid 曲线几乎是一条直线，斤率约为 0.25。如果数据全被限制在均值 0、方差 1 的标准正态分布里，那么网络经过这一层就等价于做了一个「线性变换」（`y ≈ 0.25x`）。深度学习之所以强大，就是因为它有很多「非线性」能拟合复杂的现实规律。如果全变成直线，多层神经网络就退化成了一个简单的线性回归，失去了威力。

再看 ReLU：ReLU 在负半轴是 0，正半轴是直线。如果数据总是均值为 0，意味着每次都有一半的神经元被「杀死」，这也极大限制了网络的容量。

γ 和 β 的救场就在于此：网络通过学习 γ 和 β，可以主动把分布稍微推移一点、拉宽一点。比如网络可能会学到：「把这批数据往右平移一点（`β = 2`），让它稍微进入一点 Sigmoid 的弯曲部分，这样既能得到非线性，又不至于太靠右掉进饱和区。」这就是 γ 和 β 的精妙之处：它们负责在「防止梯度消失」和「保持非线性表达能力」之间，找到一个最完美的平衡点。

**(c) 「回到原样」是网络自主的选择，而不是被迫的**

前面说的极端情况确实能把数据还原回 BN 之前的样子。但这和「没有 BN」有本质区别：没有 BN 时，数据变成那样是**失控的、被迫的**，大概率是烂分布；有 BN 时，数据变成那样是因为网络经过千百万次迭代，**主动发现**那个分布就是对当前任务最有利的分布。实践中，网络学到的 γ 和 β 几乎永远不会刚好等于原始分布的参数——它们会收敛到一个比原始分布更好的位置。

**用一个比喻串起来**：假设你在一辆飞驰的汽车上要用方向盘画一个完美的圆。没有 BN 时，车上坐了 10 个抢方向盘的人（前面的层），大家都在乱拽，车子直接冲出赛道（梯度消失）。只做归一化，等于把方向盘死死焊在正中间——车子虽然不会冲出赛道，但只能走直线，没法画圆（失去非线性）。完整的 BN（归一化 + γβ），等于先把其他 10 个人的手全部打断（归一化切断影响），然后把方向盘交给唯一一个经验老到的老司机（γ 和 β）——老司机可以根据弯道需要平稳地打方向盘，完美过弯。

**γ 和 β 的初始化策略**

在 PyTorch、TensorFlow 等所有主流框架中，BN 层的默认初始化是 **`γ = 1, β = 0`**。这里面蕴含着严谨的设计逻辑。

当 `γ = 1, β = 0` 时，公式 `y_i = γ · x_hat_i + β` 就退化成了 `y_i = x_hat_i`——**在模型刚开始训练的第一步，BN 层只做归一化，不做任何附加的缩放和平移**。此时输入给下一层激活函数的数据是 100% 纯正的「均值为 0，方差为 1」的分布。这确保了在训练初期，梯度既不会爆炸也不会消失，让模型站在最稳固的起跑线上开始学习。

随着训练的推进（Loss 开始下降，梯度开始回传），网络如果觉得 (0, 1) 的分布限制了表达能力，它就会像微调旋钮一样，**慢慢地**把 γ 从 1 移开、把 β 从 0 移开。这种**由稳定过渡到灵活**的渐进式学习策略，是保证深度网络收敛的关键。

**进阶技巧：Zero-γ 初始化**：虽然默认是 `γ = 1`，但在一些极深的残差网络中，会把每个残差块里**最后一个 BN 层的 γ 初始化为 0**。如果 `γ = 0` 且 `β = 0`，那么这个 BN 层的输出 `y_i` 全都是 0，残差块的公式 `Output = F(x) + x` 就只剩下 `Output = x`（纯短接跳线）。这相当于在训练初期，100 层的深层网络被「伪装」成了一个非常浅的网络（信息直接无损穿透），极大地降低了初期训练的难度。等模型稳住阵脚后，γ 再慢慢从 0 变大，深层网络的能力才逐步被释放出来。这个技巧在 §12.2(b) 讨论 ResNet 「让 F 归零 = 恒等映射」时有更详细的说明。

#### 11.2.1 推理（测试）阶段的做法

模型训好后去预测新数据时，通常一个一个样本输入（Batch Size = 1），这时候没办法计算「批次均值」和「批次方差」——只有一个样本，均值就是它自己，方差为 0，归一化没有意义。

解决方案是**使用全局统计量**：训练过程中，BN 层在后台会悄悄维护一个**全局均值（Running Mean）**和**全局方差（Running Variance）**，通常使用**指数移动平均（EMA）**来更新：

```
全局均值 = (1 - momentum) × 全局均值 + momentum × 当前批次均值
全局方差 = (1 - momentum) × 全局方差 + momentum × 当前批次方差
```

在 PyTorch 中，`BatchNorm2d` 默认 `momentum=0.1`，意味着每个批次的统计量以 10% 的权重融入全局值。训练结束时，全局均值/方差已经稳定地反映了整个训练集的统计特性。

**推理时**：直接用训练积累下来的全局均值和全局方差代入第 3 步进行归一化，γ 和 β 则使用训练最终确定下来的固定值。这就是为什么推理前必须调用 `model.eval()`——它会让 BN 层切换到使用全局统计量而非当前批次统计量的模式。忘了调 `model.eval()` 是深度学习新手最常踩的一个坑：模型在训练集上表现很好，推理时结果不稳定甚至很差——就是因为 BN 还在用单个样本的「批次」统计量。

#### 11.2.2 图像处理中 BN 的特殊性（Spatial BN）

在图像卷积网络中，数据的形状是 `(B, C, H, W)`，BN 是以**通道（Channel）**为单位来进行统计的。

对于某一个特定的通道 c，均值和方差**不仅在 Batch（B）维度上计算，还在空间维度（H, W）上计算**——相当于对 `B × H × W` 个像素点求一个均值和一个方差。这也叫**Spatial Batch Normalization**。

这意味着：如果有 256 个通道，就会有 **256 个 γ 和 256 个 β**——每个通道各自独立地学一组缩放和偏移。PyTorch 的 `BatchNorm2d(256)` 里的参数量就是 `2 × 256 = 512`（256 个 γ + 256 个 β），再加上 256 个 running_mean 和 256 个 running_var（不可学，只是统计量）。

**为什么是按通道而不是按像素或按整个特征图**：因为同一个通道内的所有空间位置共享同一组卷积核参数（这是 §1.2 平移不变性的直接要求），所以它们的输出分布应该相似——用同一组统计量来归一化是自然的。不同通道检测不同的特征（边缘、颜色、纹理），分布差异很大，所以必须各自独立归一化。

### 11.3 为什么规范化要在激活函数前面

BN 的位置是「Conv → BN → 激活」这个顺序，即**规范化发生在激活函数之前**。这个设计有几层原因，从表层到本质：

**表层原因：让激活函数工作在「有梯度」的区域**

拿 sigmoid 举例最直观。sigmoid 的梯度 `σ'(x) = σ(x)(1 - σ(x))`：`x = 0` 时梯度最大 = 0.25；`x = ±4` 时梯度 ≈ 0.018；`x = ±10` 时梯度 ≈ 10⁻⁴（彻底死了）。如果一个卷积层的输出（pre-activation）分布是均值 5、方差 20，那大部分值都落进了 sigmoid 的**饱和区**，梯度几乎为零——反向传播时这一层就变成了「梯度黑洞」。这就是深层 sigmoid 网络训不动的经典原因。BN 放在 sigmoid **之前**，把分布拉到 `(0, 1)`，值大部分落在 `[-2, 2]` 区间，正好是 sigmoid 梯度最大的区域。如果 BN 放在 sigmoid **之后**，sigmoid 已经把输出压到了 `(0, 1)`，饱和已经发生、梯度已经消失了，再规范化只是把 `(0, 1)` 里的值挪一挪，救不回来。

**更深一层：ReLU 也需要，只是原因不同**

ReLU 没有饱和问题，但有**死区（dead ReLU）**问题：如果 pre-activation 分布均值是 -5、方差 1，那几乎所有神经元的输出都是 0，反向传播时这些神经元的梯度也是 0——**永久死亡**。BN 放在 ReLU 之前，保证 pre-activation 均值大约 0、方差 1，那么**大约一半的神经元会被激活**（正的通过，负的被截断），网络才有健康的稀疏性。如果 BN 放在 ReLU **之后**：ReLU 的输出总是非负的，对一个非负分布做规范化会把一半的值强行推成负数，破坏了 ReLU 精心制造的「非线性 + 稀疏性」结构，而且对有硬边界的分布做 BN 训练不稳定（二阶矩估计更不可靠）。

**更本质的一层：`γ、β` 可学的意义**

BN 完整公式 `y = γ · normalize(x) + β` 里，`γ` 和 `β` 让网络**有能力抵消掉 BN 的效果**——理论上如果 `γ = √Var, β = E[x]`，就把 BN 完全撤销了。这个「网络自己决定要不要规范化」的能力，是 BN 相比 whitening 之类固定操作的关键优势。**只有当 BN 作用在线性变换的输出上，这个 `γ · x + β` 才能被下一个线性操作「吸收」**：在 `Conv → BN → ReLU` 里，BN 后紧接着的是 ReLU 再进入下一个 Conv，`γ, β` 的作用可以被后续权重完全表达；而 `Conv → ReLU → BN` 的话，BN 的 `γ, β` 之后是非线性 ReLU，不能被吸收，网络的等价类被破坏了。

**关于 Pre-activation ResNet 的看似矛盾**：ResNet-v2 里改成了 `BN → ReLU → Conv`，看起来 BN 反而在 ReLU 前面了。但关键还是「BN 作用在什么之上」——v2 里 BN 规范化的是「上一个残差块的加法输出（`x + F(x)`）」，本质仍是一个线性组合的结果。两种版本里，BN 都是紧跟在一个线性操作之后。**「BN 在激活函数前」的真正含义不是位置本身，而是「BN 作用在线性输出、送进非线性之前」**。

### 11.4 使用示例

```python
# 卷积层后
block = nn.Sequential(
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),         # 参数是通道数
    nn.ReLU(inplace=True),
)

# 全连接层后
mlp = nn.Sequential(
    nn.Linear(1024, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
)
```

### 11.5 效果与相关变体

- 可以使用**更大的学习率**、训练更**稳定**
- 有一定**正则化**效果（因为每个 batch 的统计量都有噪声）
- 使**很深的网络**训得动 —— 这为 ResNet 铺了路

**相关变体**（了解即可）：LayerNorm（[[Transformer]] 用）、GroupNorm（batch 小时用）、InstanceNorm（风格迁移用）。

---

## 12. ResNet：残差网络（重点）

He Kaiming et al., 2015。**深度学习历史上最重要的架构之一**，2015 ImageNet 冠军，被引用超过 20 万次。它让「训 100+ 层的网络」变成可能，深刻影响了此后所有神经网络的设计（包括 [[Transformer]]）。

### 12.1 为什么需要 ResNet：函数类的嵌套

先想一个反直觉的事实：**把网络加深，训练误差反而会变大**（不是过拟合，是训练集本身的 loss 上升）。这被称为**退化问题（degradation）**。

**用「函数类」的视角理解这件事**：

一个网络结构能表示的所有可能函数，构成一个集合，叫「函数类」。比如：

- **F_浅** = 「6 层 CNN」能表达的所有函数
- **F_深** = 「12 层 CNN」能表达的所有函数

我们真正想要的最优函数 `f*`（现实中的理想模型）通常不在任何一个函数类里，只能在函数类中找一个最接近它的替代品——记 F_浅 中最接近 `f*` 的那个为 `f*_浅`，F_深 中最接近的为 `f*_深`。

**关键问题**：把网络加深（F_浅 → F_深），能保证 `f*_深` 比 `f*_浅` 更接近 `f*` 吗？

答案是——**只有当 F_浅 ⊆ F_深（F_深 包含 F_浅 里的每一个函数）时才能保证**。这种情况叫「**函数类嵌套（nested function classes）**」。

- **嵌套时**：F_深 里至少能找到一个和 `f*_浅` 一样的函数，所以 `f*_深` 最坏也和 `f*_浅` 一样好，通常更好。加深有利无害。
- **不嵌套时**：F_深 虽然「更大」，但方向可能歪了——里面可能根本没有 `f*_浅` 那么好的函数。加深反而变差。

**普通深层网络恰好不满足嵌套**：12 层网络要复现 6 层网络的解，需要让新加的 6 层学成「恒等映射」（`f(x) = x`），但一堆卷积 + BN + ReLU 要精确凑出恒等映射非常困难——所以就出现了退化现象。

**ResNet 的核心思想**：把每一层设计成 `y = F(x) + x` 的形式。只要 F(x) 学成 0（把卷积权重推向 0），整个块就退化成恒等映射——这**极其容易**。于是深层残差网络天然满足 F_浅 ⊆ F_深，**加深永远不会变差**。

### 12.2 深入理解「让 F 归零 = 恒等映射」

上一节说「让 F(x) 学成 0 就得到恒等映射，这极其容易」，这句话压缩了三层意思，值得展开：

**(a) F(x) = 0 怎么就让整个块变成恒等映射**

残差块 `y = ReLU(F(x) + x)`。假设 F 分支里两个卷积层的权重**全部设为 0**：第一个 Conv `W₁ · x = 0`（因为 W₁ = 0），BN(0) 初始 β=0 时输出 0，ReLU(0) = 0，第二个 Conv `W₂ · 0 = 0`，第二个 BN(0) = 0。所以 `F(x) = 0`，不管输入 x 是什么。于是 `y = ReLU(0 + x) = ReLU(x)`——如果 x 来自上一层的 ReLU 输出（本身就非负），那 `ReLU(x) = x`，这个残差块什么都没做，输入直接透过去了。

（严格的恒等在 v1 结构下只在 x ≥ 0 时成立；Pre-activation v2 把 ReLU 提前，恒等更彻底。参见 §12.8。）

**(b) 为什么「把权重推向 0」极其容易**

深度学习的默认设置几乎是**主动帮你**把权重推向 0：

1. **初始化就靠近 0**：Kaiming init 采样自 `N(0, 2/fan_in)`，方差 O(1/n)。权重一开始就在 0 附近晃悠。
2. **权重衰减主动往 0 拉**：训练时的 `SGD(..., weight_decay=5e-4)` 就是 L2 正则，每步把权重乘以一个略小于 1 的系数。这本身就在往 0 拉。
3. **没用的残差块会自然衰减**：如果某个残差块对最终 loss 没有贡献，它的梯度就很小；加上 weight decay 一直拉，权重自然衰减到近 0。
4. **更极端的技巧：Zero-init**：后续改进（Fixup、Zero-γ trick）把每个残差块**最后一个 BN 的 γ 初始化为 0**。因为 BN 输出 = `γ · normalize(x) + β`，γ = 0 时输出恒为 β（也初始化为 0），所以 F(x) = 0 **从训练第 0 步就成立**。整个网络初始化时就是一堆恒等块的堆叠，训练过程「按需激活」需要的块。这个技巧在超深网络（1000+ 层）里几乎是标配。

**(c) 对比：普通网络为什么做不到**

同样一个「12 层普通 CNN」想复现「6 层普通 CNN」的行为，需要让新加的 6 层复合起来等于恒等映射。让一个普通卷积块（`Conv → BN → ReLU`）变成恒等映射需要：Conv 的权重恰好是恒等映射的卷积核（3×3 卷积核中心 1、周围 0，多通道还要精准匹配）；BN 的 γ、β 恰好抵消 Conv 的效应；ReLU 只对非负输入是恒等，负数会截断——还得保证输入永远非负。**每一条都是极其特殊、精细的权重配置**。随机初始化根本碰不到，梯度下降也不会主动往这个方向走（因为途中任何一步偏离都会让 loss 上升，形成陷阱）。

打个比方：**ResNet 让块变成恒等**就像把音量旋钮拧到 0——一个动作；**普通网络让块变成恒等**就像让一整支乐队精确合奏出「静默」——每个乐器的音波要刚好互相抵消。一个方向明确、操作简单；一个复杂脆弱、稍偏就崩。这就是「极其容易」和「几乎不可能」的差别。

**(d) 由此推出「深层函数类 ⊇ 浅层函数类」**

设 N_浅 是一个已经训好的 6 层 ResNet 实现的函数 `f`；N_深 是在 N_浅 后面再加 6 个残差块得到的 12 层 ResNet。**构造**：让 N_深 的前 6 个残差块的权重 = N_浅 的权重；让后加的 6 个残差块的 F 分支权重全设为 0（恒等映射）。那么 N_深(x) = 恒等 ∘ ... ∘ 恒等 ∘ N_浅(x) = N_浅(x) = f(x)。因此，**任何 6 层 ResNet 能表示的函数，12 层 ResNet 也一定能表示**。用集合语言：F_浅 ⊆ F_深。这就是嵌套。

而普通深层网络没这个保证：让新加的 6 个普通卷积块合成恒等的那组权重虽然理论上存在，但根本训不到那里。所以 `f*_普通深` 比 `f*_普通浅` 更差的情况经常发生——这就是**退化现象**的根本原因。

**一句话**：ResNet 用「加法 + 让 F 归零」的机械式设计，把「保留浅层网络能力」从一个**需要精心学习的目标**变成了一个**默认发生的行为**。这个小改动，把「更深 = 更强」从愿望变成了数学保证。

### 12.3 残差块的结构
![[Pasted image 20260826170003.png]]

![[Pasted image 20260826170120.png]]
假设想学的理想映射是 `f(x)`，普通网络直接拟合 `f`；ResNet 拟合的是**残差** `g(x) = f(x) - x`，然后输出

```
f(x) = g(x) + x
```

当理想映射接近恒等时，只需 `g ≈ 0`，即把残差分支的权重推向 0 就行——这比让一堆卷积层去凑出恒等映射容易得多。

**基本残差块（BasicBlock）**（用于 ResNet-18 / 34）：

```
输入 x
 ├── 3x3 conv → BN → ReLU ──┐
 │                          │
 │   3x3 conv → BN          │
 │                          │
 └──────────────────────────┼── + ── ReLU ── 输出
        （恒等旁路）
```

**当通道数或空间尺寸需要变化时**（图中虚线 shortcut）：旁路要通过一个 **1×1 conv**（可能带 stride）把 `x` 的形状调整到与主分支一致，才能相加。这个组件叫 **downsample**，详见 §12.5。

### 12.4 残差块的通用 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """ResNet-18/34 使用的基础残差块。"""
    expansion = 1  # 输出通道 = out_channels * expansion

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        # 当输入输出形状不一致时，用这个把 x 变形后再相加
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity           # 残差相加
        return F.relu(out, inplace=True)
```

**几点重要细节**：
- 卷积层后跟 BN 时通常设 `bias=False`（BN 的 β 会替代 bias）
- ReLU 放在残差相加**之后**（这是原版 ResNet-v1 的做法）
- `downsample` 是 shortcut 上的 `1×1 conv + BN`

### 12.5 downsample 详解

残差块的核心公式 `y = F(x) + x` 要成立，**两个张量的形状必须完全一致**（同样的通道数、同样的空间尺寸 H×W）。否则 PyTorch 会直接报 shape mismatch 错误。

问题在于，ResNet 的整体设计是「随着深度增加，空间尺寸下采样、通道数翻倍」：

| stage | 输入 | 输出 |
|---|---|---|
| layer2 第一块 | 64 × 56 × 56 | **128 × 28 × 28** |
| layer3 第一块 | 128 × 28 × 28 | **256 × 14 × 14** |
| layer4 第一块 | 256 × 14 × 14 | **512 × 7 × 7** |

看 layer2 的第一块：输入是 `(64, 56, 56)`，主分支 F 通过 `stride=2` 卷积把它变成了 `(128, 28, 28)`。但旁路 x 还是 `(64, 56, 56)`——通道数和空间尺寸**都不匹配**，没法相加。这时候就需要 **downsample**：一个专门作用在旁路 x 上的模块，把 x 的形状调整成和 F(x) 一致，让加法能进行。

**downsample 具体做什么**：

```python
if stride != 1 or self.in_planes != planes * block.expansion:
    downsample = nn.Sequential(
        nn.Conv2d(self.in_planes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
    )
```

它是一个 `Conv2d + BatchNorm2d` 的小组合：
- **1×1 卷积改通道数**：输入通道 `self.in_planes`（比如 64），输出通道 `planes * expansion`（比如 128）。1×1 卷积在通道维度上做线性映射，不改变空间尺寸，是最便宜的改通道方式。
- **stride 改空间尺寸**：如果 `stride=2`，卷积滑动步长为 2，输出的 H 和 W 各变成原来的一半。刚好和主分支 F 里第一个 3×3 卷积的下采样对齐。
- **BatchNorm 保持数值稳定**：既然主分支每层后面都有 BN，旁路也加一个 BN，让相加的两边数值尺度相当。

**什么时候需要 downsample**：`_make_layer` 里的判断条件是 `if stride != 1 or self.in_planes != planes * block.expansion`。两种情况：
- **空间尺寸变化**（`stride != 1`）：每个 stage 的第一块要下采样一半，比如 layer2/3/4 的第一块 stride=2。
- **通道数变化**（`in_planes != planes * expansion`）：每个 stage 的第一块要改通道数。对 Bottleneck（expansion=4）更常见，因为输入通道和输出通道倍数关系更容易不匹配。

**其他情况**：一个 stage 内部除第一块外，输入输出形状完全一致，`downsample = None`，旁路直接透过 x。

**downsample 会不会破坏「恒等映射」的性质**：在形状必须变化的位置，本来就无法维持严格的恒等映射（形状都变了，谈何恒等）。这时候的目标是「让旁路做最简单、最接近恒等的一个形状调整」——1×1 卷积是所有能改通道/下采样的操作里**最简单**的（参数最少、无非线性）。

- **stage 内部的块**（downsample = None）：真正的恒等旁路，「归零 F 就恢复浅层网络」严格成立。
- **stage 交界处的第一块**（有 downsample）：形状必须变，旁路是一个「最接近恒等的线性投影」。

这个区别在 ResNet 原论文里被称为 **identity shortcut**（真恒等）vs **projection shortcut**（投影），何恺明论文的表格里做过对比实验：多数位置用 identity，少数必须变形的位置用 projection，这个组合效果最好、参数最省。

### 12.6 完整 ResNet-18 实现

![[Pasted image 20260826170427.png|325]]
**架构表**（针对 ImageNet 输入 3×224×224）：

| 阶段 | 输出尺寸 | 结构 |
|---|---|---|
| conv1 | 64 × 112 × 112 | 7×7 conv, 64, stride=2 → BN → ReLU |
| maxpool | 64 × 56 × 56 | 3×3 max pool, stride=2 |
| layer1 | 64 × 56 × 56 | 2 × BasicBlock(64) |
| layer2 | 128 × 28 × 28 | 2 × BasicBlock(128), 第一块 stride=2 |
| layer3 | 256 × 14 × 14 | 2 × BasicBlock(256), 第一块 stride=2 |
| layer4 | 512 × 7 × 7 | 2 × BasicBlock(512), 第一块 stride=2 |
| avgpool | 512 × 1 × 1 | 全局平均池化 |
| fc | num_classes | Linear(512, num_classes) |

参数量约 11.7M。

**关于「ResNet-18」中「18」怎么数出来的**：网络深度这个词其实没有统一定义，不同语境下意思不一样。主流数法是——**数带可学习参数的层**：Conv 层（不管是 3×3、1×1 还是 7×7，都算 1 层）和 Linear 层都算；BN（虽然有 γ/β 参数但不算独立层）、ReLU、Pool、Dropout、Flatten 都不算。按这个规则数 ResNet-18：

```
1.  开头 7x7 conv                              → 1
    maxpool  (不算)
2.  layer1: 2 × BasicBlock，每块 2 个 3x3 conv → 4
3.  layer2: 2 × BasicBlock                     → 4
4.  layer3: 2 × BasicBlock                     → 4
5.  layer4: 2 × BasicBlock                     → 4
6.  末尾 Linear                                → 1
    ────────────────────────────────
    合计:  1 + 4×4 + 1 = 18
```

**注意 downsample 里的 1×1 卷积不算在这 18 里**——这是约定俗成的例外，因为 downsample 只是形状适配、不算主要计算路径。

**块、stage、层三个概念的区分**：

| 术语            | 含义                            | ResNet-18 里的数量           |
| ------------- | ----------------------------- | ------------------------ |
| **块（block）**  | 一个残差单元（BasicBlock/Bottleneck） | 8（每 stage 2 块 × 4 stage） |
| **层组（stage）** | 特征图分辨率相同的连续块                  | 4（layer1~4）              |
| **层（layer）**  | 有可学习参数的单个变换（conv/linear）      | 18                       |

torchvision 代码里把 stage 也叫 `self.layer1, layer2, ...`——这个命名和「深度里数的 layer」不是一回事，容易搞混。

**代码实现**：

```python
class ResNet(nn.Module):
    def __init__(self, block, layers: list[int],
                 num_classes: int = 1000, in_channels: int = 3):
        super().__init__()
        self.in_planes = 64
        # Stage 0: 首层
        self.conv1   = nn.Conv2d(in_channels, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 4 个 stage
        self.layer1 = self._make_layer(block,  64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * block.expansion, num_classes)

        # 初始化：卷积用 Kaiming，BN 权重初始化为 1，bias 为 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def _make_layer(self, block, planes: int, num_blocks: int,
                    stride: int) -> nn.Sequential:
        downsample = None
        # 当形状变化时，构造 1x1 downsample 分支
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet18(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channels)


def resnet34(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, in_channels)
```

### 12.7 Bottleneck 块与 ResNet-50/101/152

对更深的网络（50 层以上），使用 3×3 卷积做主干太贵。ResNet 引入了 **Bottleneck** 结构：

```
输入 x (in_c)
 ├── 1x1 conv (in_c → planes)          BN → ReLU     # 降维
 │   3x3 conv (planes → planes)         BN → ReLU     # 主运算
 │   1x1 conv (planes → planes*4)       BN            # 升维
 └────────────────────────────────────── + → ReLU
```

一个 bottleneck 块的输出通道是「中间通道 × 4」，所以 `expansion = 4`。参数比同感受野的两层 3×3 少得多。

```python
class Bottleneck(nn.Module):
    """ResNet-50/101/152 使用的瓶颈残差块。"""
    expansion = 4

    def __init__(self, in_channels: int, planes: int,
                 stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        out_channels = planes * self.expansion
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_channels, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return F.relu(out, inplace=True)


def resnet50(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, in_channels)

def resnet101(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, in_channels)

def resnet152(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, in_channels)
```

**ResNet 家族**：

| 模型 | Block | 每 stage 块数 | 参数量 | ImageNet Top-1 |
|---|---|---|---|---|
| ResNet-18 | Basic | [2,2,2,2] | ~11.7M | ~70% |
| ResNet-34 | Basic | [3,4,6,3] | ~21.8M | ~73% |
| ResNet-50 | Bottleneck | [3,4,6,3] | ~25.6M | ~76% |
| ResNet-101 | Bottleneck | [3,4,23,3] | ~44.5M | ~77% |
| ResNet-152 | Bottleneck | [3,8,36,3] | ~60.2M | ~78% |

### 12.8 Pre-activation ResNet（v2）：梯度传播路径的干净化

原版 ResNet（v1）的块顺序是 `Conv → BN → ReLU`；ResNet-v2（He et al., 2016 后续论文 *Identity Mappings in Deep Residual Networks*）改成 `BN → ReLU → Conv`（pre-activation），让梯度传播路径更「干净」，非常深的网络（1000+ 层）也能训得动。为什么这么做梯度更干净，需要看反向传播的数学。

**两个版本到底差在哪**：

v1 原版：`y = ReLU(F(x) + x)`，F 分支是 `Conv → BN → ReLU → Conv → BN`；
v2：`y = F(x) + x`，F 分支是 `BN → ReLU → Conv → BN → ReLU → Conv`。

**核心差异只有一处**：v1 在加法**之后**还有一个 ReLU；v2 加法**之后什么都没有**，直接作为下一个块的输入。

**前向传播看：递归关系变干净了**

v2 里，把 L 个残差块串起来：`x_{l+1} = x_l + F_l(x_l)`。展开成闭式：

```
x_L = x₀ + Σ_{i=0}^{L-1} F_i(x_i)
```

**任意层的输出 = 初始输入 + 前面所有残差项的累加**。这个恒等式**精确成立**，是干净的加法结构。

v1 里就不行了：`x_{l+1} = ReLU(x_l + F_l(x_l))`。外面这个 ReLU 破坏了加法关系。没法把 `x_L` 展开成 `x₀ + Σ F_i(x_i)` 这种形式，因为每一步的 ReLU 都非线性地截断了一部分信号。

**反向传播看：梯度有了一条「高速公路」**

v2 里，对 `x_L = x₀ + Σ F_i(x_i)` 求 `x₀` 的梯度：

```
∂Loss/∂x₀ = ∂Loss/∂x_L · ∂x_L/∂x₀
         = ∂Loss/∂x_L · (1 + ∂/∂x₀ Σ F_i(x_i))
         = ∂Loss/∂x_L · 1  +  ∂Loss/∂x_L · (中间项)
           ────────────
           关键：这个 1
```

**这个 "1" 是核心**——它意味着 `∂Loss/∂x_L` 会**原封不动、无衰减、无门控**地直接传到 `∂Loss/∂x₀`。相当于给梯度开了一条从最深处直达最浅处的高速公路，中间不设收费站、不限速、不检查。

v1 里就不一样。`x_{l+1} = ReLU(x_l + F_l(x_l))`，链式法则求梯度：

```
∂x_{l+1}/∂x_l = ReLU'(x_l + F_l(x_l)) · (1 + ∂F_l/∂x_l)
                ─────────────────────
                这里多了一个 0 或 1 的 mask
```

`ReLU'` 的值只能是 0（输入为负时）或 1（输入为正时）。这意味着**梯度的「高速公路」上每一层都有一个门**，门开着才能过（乘 1），门关着直接堵死（乘 0）。跨 L 层反传梯度：

```
∂Loss/∂x₀ = ∂Loss/∂x_L · ∏_{l=0}^{L-1} [ ReLU'(...) · (1 + ...) ]
                          ─────────
                          L 个 0/1 mask 连乘
```

L 个 mask 连乘的意思是：**只要中间任何一层的 ReLU 关掉了（输入为负），整条梯度路径就被切断**。即使 mask 都是 1，v1 也没有 v2 那个「无条件保留的加号 1」，梯度天然带一层「非线性衣服」，不再是纯粹的恒等传播。

**为什么这对 1000+ 层尤其致命**：

假设每一层的 ReLU 有 90% 的概率被激活（mask = 1），听起来很高：

- **10 层**：`0.9¹⁰ ≈ 0.35`，剩 35% 的梯度信号能传到最浅层。还行。
- **100 层**：`0.9¹⁰⁰ ≈ 2.7 × 10⁻⁵`，梯度信号衰减到十万分之三。基本训不动。
- **1000 层**：`0.9¹⁰⁰⁰ ≈ 10⁻⁴⁶`。彻底死透。

这是**指数级衰减**——只要每一层引入哪怕微小的信号损失，堆叠 1000 层后剩下的信号趋近于 0。v1 训 200 层左右还能勉强，训到 1000 层就崩了。v2 完全绕开这个问题：那个「无条件的 1」不会被 mask，跨多少层都一样。何恺明在 v2 论文里做了实验，用 v2 结构成功训练了 1001 层的 ResNet，训练误差比 152 层还低——这在 v1 上是完全不可能的。

**除了梯度之外，v2 还有几个附带好处**：

- **BN 作用在「原始输入」上，而不是「加法结果」上**：v1 里 BN 作用在 conv 输出上，加法发生在 BN 之后——加法后的分布已经不是标准化的了。v2 里每个块开头是 `BN → ReLU`，BN 作用在前一块的加法输出上，这样每个 conv 看到的输入都是标准化的，BN 的作用更充分。
- **加法输出不受非线性扭曲**：v1 里加法结果被 ReLU 截断为非负，堆很多层时信号会不断被「只留正数」累积，出现分布漂移。v2 里加法结果既有正也有负，直接送入下一块的 BN 重新标准化。
- **极端情况：把 F 学成 0，v2 是严格恒等映射**：v1 是 `y = ReLU(0 + x) = ReLU(x)`，只在 x ≥ 0 时是恒等；v2 是 `y = 0 + x = x`，严格恒等。v2 让「恒等映射默认」这个 ResNet 核心思想更彻底地成立。

**直观类比**：把 L 个残差块想象成 L 段接力赛的传信。v1 是每段接力员都要复述一遍消息，然后经过一个「审核员」（ReLU）决定是否放行——审核员心情不好（输入为负）就把消息毁掉；v2 是每段接力员只负责在原消息上**附加一些补充信息**，原消息永远保持完整传递。用一句话总结：**v2 让残差网络真正实现了「x 是主线，F 是补丁」的哲学——主线畅通无阻，补丁按需附加。v1 只是形式上的残差，主线上仍然堆着 ReLU 收费站；v2 把收费站全部搬进了 F 分支内部，主线彻底干净了。**

**代码实现**：

```python
class PreActBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        identity = self.downsample(out) if self.downsample is not None else x
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        return out + identity   # 注意：没有最后的 ReLU
```

这也是为什么现在几乎所有基于残差思想的架构（[[Transformer]] 的 `x + Attn(x)` 和 `x + FFN(x)`、扩散模型的 U-Net、Diffusion Policy 等）都采用类似的干净加法结构——**保护主线**是深度学习工程的一个基本原则。

### 12.9 训练脚本示例

用 CIFAR-10 训练自实现的 ResNet-18（不依赖 d2l）：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def train_resnet_on_cifar10(
    num_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # ---------- 数据 ----------
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    test_set  = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256,       shuffle=False,
                              num_workers=4, pin_memory=True)

    # ---------- 模型 / 优化器 / 学习率调度 ----------
    net = resnet18(num_classes=10, in_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ---------- 训练循环 ----------
    for epoch in range(num_epochs):
        net.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = net(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # ---------- 评估 ----------
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (net(x).argmax(1) == y).sum().item()
                total   += x.size(0)
        print(f"epoch {epoch+1:3d} | test acc {correct/total:.4f}")
```

> ⚠️ **给 CIFAR-10 一点提示**：上面 `resnet18` 的首层是 7×7 stride=2 + maxpool，对 32×32 的小图会一开始就下采样到 8×8 太狠，直接训准确率会打折。CIFAR 版 ResNet 通常把首层改成 3×3 stride=1 且去掉 maxpool（He 原论文中 CIFAR 实验就是这么做的）。

### 12.10 ResNet 小结

一句话记忆：**残差块让恒等映射成为默认，加深网络永远不会更差；1×1 卷积和 BN 是让它跑起来的两根支柱。**

设计上的四个要点：
1. **两层 3×3 conv + BN + ReLU** 作为主分支
2. **恒等旁路（skip connection）** 直接把输入加到主分支输出
3. **形状变化时用 1×1 conv shortcut**（下采样 stride=2）
4. **深层版本用 bottleneck**（1×1 降维 + 3×3 + 1×1 升维）

---

## 13. DenseNet：稠密连接网络

Huang et al., 2016。把「残差」推向极致：**每一层的输入都是它前面所有层输出的拼接**（不是相加，是 `concat`）。

设第 ℓ 层的输出是 `x_ℓ`，则：

```
x_ℓ = H_ℓ( [x_0, x_1, ..., x_{ℓ-1}] )
```

其中 `[·]` 是通道维拼接。这样特征被最大限度地复用，参数量反而更少（因为每层只需要新增少量通道，称为 **growth rate**）。

- **Dense Block**：内部所有层稠密相连，输出通道数不断增长
- **Transition Layer**：块之间用 `1×1 conv + 2×2 avg pool` 降通道、降尺寸

DenseNet 的实际使用不如 ResNet 广泛，但「稠密连接」的思想在很多分割/检测网络里出现。

```python
def dense_layer(in_c: int, growth_rate: int) -> nn.Sequential:
    return nn.Sequential(
        nn.BatchNorm2d(in_c), nn.ReLU(inplace=True),
        nn.Conv2d(in_c, growth_rate, kernel_size=3, padding=1, bias=False),
    )

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, in_c: int, growth_rate: int):
        super().__init__()
        self.layers = nn.ModuleList([
            dense_layer(in_c + i * growth_rate, growth_rate)
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)   # 拼接，不是相加
        return x
```

---

# 第三部分：附录

## A. 参数量与显存估算

在设计或选择模型时，快速估算参数量能帮判断能不能塞进显存、要多大 batch size。核心思路是「抓大头」——每种层参数量公式简单，占大头的层不多。

### A.1 每种层的参数量公式

**Conv2d**（最常见的大头）：

```
参数量 = C_in × C_out × k_h × k_w  +  C_out  (bias)
```

如果配 BN，通常 `bias=False`（bias 会被 BN 的 β 吸收），就没有最后那个 `+ C_out`。**常用简化记忆**：
- 3×3 conv：`9 × C_in × C_out`
- 1×1 conv：`C_in × C_out`
- 7×7 conv：`49 × C_in × C_out`

**Linear（全连接）**：

```
参数量 = in_features × out_features  +  out_features (bias)
```

一个 `Linear(4096, 4096)` 就有 **16.8M** 参数——所以 VGG 里那三个 FC 层贡献了绝大部分参数。

**BatchNorm**：`参数量 = 2 × num_features`（γ 和 β）。`BatchNorm2d(512)` 只有 1024 个参数，**基本可以忽略**。

**其他常见层**：

| 层 | 参数量 | 备注 |
|---|---|---|
| Pooling (Max/Avg) | 0 | 无参数 |
| ReLU/Sigmoid/GELU | 0 | 无参数 |
| Dropout / Flatten | 0 | 无参数 |
| Embedding | `vocab_size × dim` | NLP 里常是大头 |
| LayerNorm | `2 × dim` | [[Transformer]] 里可忽略 |
| MultiHeadAttention | `4 × dim²` | Q/K/V/O 四个投影，忽略 bias |

### A.2 抓大头的直觉

看一个 CNN 时不用把每层都算，**几个关键位置就决定了 80% 的参数量**：

- **深层（通道数大）的 3×3 conv 是绝对大头**：`3×3 conv, 512→512` = `9 × 512 × 512` = 2.36M 参数。ResNet-18 的 layer4 里有 4 个这种卷积，光这一个 stage 就是 ~9.4M——占了整个网络（11.7M）的 80%。**规律：参数量随通道数平方增长**（`C_in × C_out` 项）。所以「加深浅层」比「加深深层」便宜得多。
- **全连接层要警惕**：`Linear(N, M)` 是 `N × M`，两个数一乘很快就爆炸。VGG-16 那个 `Linear(25088, 4096)` = 103M 参数，占整个 VGG 的 74%。**这是为什么现代 CNN 都用 GAP 替代 FC**。
- **1×1 conv 便宜 9 倍**：同样通道数，1×1 conv 只有 3×3 的 1/9 参数。这就是 Bottleneck 结构的核心：用 `1×1 降维 → 3×3 → 1×1 升维` 代替 `3×3 → 3×3`，参数量能省一半以上。
- **BN / ReLU / pool 基本不用管**。

### A.3 快速估算的心算模板

**Transformer 类**（12 层 base，dim=768，FFN=4× hidden）：

```
每层参数 ≈ 12 × dim²
= 12 × 768²
≈ 7M

总参数 ≈ 层数 × 每层 + embedding
      = 12 × 7M + 30K × 768
      ≈ 84M + 23M
      ≈ 107M         (实际 BERT-base 是 110M)
```

「每层 12 × dim²」这个记忆点很有用：4 dim² 是 attention 的 Q/K/V/O，8 dim² 是 FFN 的两个 4× 扩展。

### A.4 常见模型的参考量级

| 模型 | 参数量 | 说明 |
|---|---|---|
| LeNet | ~60K | 5 万 |
| MobileNetV2 | 3.5M | 移动端 |
| EfficientNet-B0 | 5.3M | |
| ResNet-18 | 11.7M | 常用基线 |
| ResNet-50 | 25.6M | 视觉迁移学习标配 |
| ViT-Base/16 | 86M | 见 [[ViT]] |
| [[BERT]]-base | 110M | |
| ResNet-152 | 60M | |
| VGG-16 | 138M | FC 层胖 |
| GPT-2 small | 117M | |
| ViT-Large/16 | 307M | |
| Llama-2 7B | 7B | 大模型入门 |
| Llama-2 70B | 70B | |

**三个锚点**：**10M 级**（轻量视觉 backbone）、**100M 级**（ResNet-50 到 ViT-Base、BERT-base）、**1B+ 级**（LLM、大视觉基础模型）。

### A.5 在 PyTorch 里精确统计

```python
def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total, trainable = count_params(resnet18())
print(f"Total: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M")
# Total: 11.69M | Trainable: 11.69M
```

**分层看细节推荐 `torchinfo`**：

```python
from torchinfo import summary
summary(resnet18(), input_size=(1, 3, 224, 224))
```

会打印出每层的参数量、输出形状、FLOPs，一眼看出谁是大头。冻结部分层做微调时也用它检查。

### A.6 从参数量估算显存占用

参数量直接决定显存底线。**推理时（fp32）**：`显存 ≈ 4 × 参数量 (bytes)`。100M 参数 × 4 bytes = 400 MB。

**训练时（fp32 + Adam）**：

```
显存 ≈ 4 × 参数量           (权重)
     + 4 × 参数量           (梯度)
     + 8 × 参数量           (Adam 的 m 和 v 两个动量)
     + 激活值 (batch 相关)
──────────────────────────
     ≈ 16 × 参数量 + 激活
```

100M 参数训练时**光权重相关就占 1.6GB**。这就是为什么 Llama 7B 用 fp32 训练需要 ≥ 100GB 显存（不含激活）。

**能真正塞进 24GB 的模型上限**（用 Adam 全量微调，激活占一半）：
- fp32：约 **500M~1B 参数**
- fp16（混合精度）：约 **1B~2B**
- + LoRA / 冻结大部分层：可以到 **7B**

对机器人操作任务，用 ResNet-18/50 或 ViT-Base 作为视觉 encoder 完全够用，显存主要花在 batch size 和序列长度（如果做 diffusion policy 的 denoising steps）上，而不是模型参数本身。

---

## B. 演进主线与横向对比

### B.1 从 LeNet 到 DenseNet 的演进主线

| 模型 | 年份 | 关键贡献 | 深度 |
|---|---|---|---|
| LeNet | 1998 | 卷积 + 池化 + FC 的雏形；证明可行 | 5 |
| AlexNet | 2012 | ReLU、Dropout、数据增强、GPU 训练 | 8 |
| VGG | 2014 | **块的思想**；3×3 小卷积堆叠 | 11-19 |
| NiN | 2013 | 1×1 conv；**全局平均池化替代 FC** | ~4 blocks |
| GoogLeNet | 2014 | Inception：**多分支并行**；1×1 瓶颈降维 | 22 |
| Batch Norm | 2015 | 让深层网络训得起来的关键技巧 | — |
| **ResNet** | 2015 | **残差连接**：加深不再退化；被后续所有架构继承 | 18-152+ |
| DenseNet | 2016 | 稠密连接：特征最大化复用 | 121-264 |

### B.2 三种建模思路的对比

|  | 相加/相连方式 | 直觉 |
|---|---|---|
| **VGG** | 顺序堆叠 | 更深更好，但会遇到退化问题 |
| **ResNet** | `y = f(x) + x` | **默认恒等**，只学差量 |
| **DenseNet** | `y = [x, f(x)]` | 保留所有历史特征、避免丢信息 |

---

## C. 通用的 CNN 设计原则

1. **早期用大 stride 或 pool 快速降空间维度**（图像很大，早点降）
2. **通道数随空间尺寸减小而翻倍**（保持每层的表达能力大致平衡；三条理由的详细讨论见 §4.4）
3. **BN + ReLU + Conv 是标准三件套**
4. **深层网络必须有残差连接**（否则梯度会有问题）
5. **末端用 GAP + Linear**，避免巨大 FC 层
6. **1×1 卷积** 用于通道升降维和跨通道混合，是几乎所有现代块的组成部分

## D. 学习路径建议

1. **必须自己实现一遍**：LeNet、BasicBlock、Bottleneck、Inception，写完就通了大半
2. **读原论文**：AlexNet、VGG、Inception v1、BN、ResNet（尤其 ResNet 的两篇：v1 与 pre-activation v2）
3. **看 torchvision 源码**（`torchvision.models.resnet`）：工业级实现，非常干净
4. **在自己的任务上跑通**：迁移学习是最快的验证方式

---

## 相关笔记

- [[Linear neural networks]]
- [[Transformer]]
- [[ViT]]
- [[BERT]]
