---
authors: Dosovitskiy et al.
paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
status: studied
tags:
  - paper
  - vision
  - transformer
  - classification
year: 2021
---

# Vision Transformer (ViT)

## 一、核心思想

ViT 的本质就是一句话：**把图像伪装成 token 序列，喂给 Transformer encoder 做分类**。

NLP Transformer 的输入是 `[B, L, d]` 的 token 序列（L 个词，每个词 embedding 成 d 维向量），经过若干个 self-attention 层，输出 `[B, L, d]`。ViT 做的事就是把一张图像变成"L 个词"——把 32×32 图像切成 4×4 的小方块（patch），每个 patch 当成一个"词"。32×32 切出 (32/4)×(32/4)=64 个 patch，每个 patch 是 4×4×3=48 维的小图块，再通过一个线性层投影到 d 维（比如 256 维）。于是图像就变成了 `[B, 64, 256]` 的序列——和 NLP 的 token 序列一模一样的形状。

### 为什么这么设计

在 ViT 之前，CV 几乎全是 CNN。CNN 内置两个归纳偏置（inductive bias）：局部性（邻近像素相关）和平移不变性（卷积核共享）。这两个偏置在数据少时是优势，但限制了模型表达能力。ViT 的赌注是：**只要数据足够多，模型自己能从 attention 里学到这些归纳偏置，并且能学到 CNN 学不到的全局长程关系**。原论文（Dosovitskiy et al. 2021）在 JFT-300M 上预训练后碾压 ResNet，证明了这点。

但反过来，在小数据集（CIFAR-10、ImageNet-1K）上从头训 ViT 是会输给同等规模 CNN 的。

---

## 二、整体数据流（带形状）

以 CIFAR-10（32×32×3，10 类）为例，patch_size=4，embed_dim=256：

```
输入图像 [B, 3, 32, 32]
    ↓ PatchEmbedding：Conv2d(kernel=4, stride=4)
patch 序列 [B, 64, 256]              ← 64 个 patch，每个 256 维
    ↓ 拼接 CLS token
[B, 65, 256]                          ← 前面多一个 [CLS]
    ↓ + 位置编码 pos_embed
[B, 65, 256]
    ↓ Dropout
    ↓ Encoder Block × N（N=6 或 12）
[B, 65, 256]
    ↓ LayerNorm
    ↓ 取出 CLS 位置：x[:, 0]
[B, 256]
    ↓ 分类头 Linear(256, 10)
logits [B, 10]
```

和 NLP Transformer encoder 比，**多的就只有最前面的 PatchEmbedding 和最后的分类头**。中间的 encoder block 完全复用。

---

## 三、各模块详解

### 3.1 Patch Embedding

**作用**：把图像切成 patch，每个 patch 投影成 embed_dim 维向量。
**输入**：`[B, 3, 32, 32]` → **输出**：`[B, 64, 256]`

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        # patch 总数 = (32/4)^2 = 64
        self.num_patches = (image_size // patch_size) ** 2

        # 关键技巧：用 Conv2d 一步完成"切 patch + 线性投影"
        # kernel_size=stride=patch_size，让卷积核刚好覆盖一个 patch、不重叠地滑动
        self.proj = nn.Conv2d(
            in_channels=in_channels,    # 3
            out_channels=embed_dim,     # 256
            kernel_size=patch_size,     # 4
            stride=patch_size           # 4
        )

    def forward(self, x):
        # x: [B, 3, 32, 32]
        x = self.proj(x)
        # 卷积后：[B, 256, 8, 8]  ← 8=32/4，相当于 8×8=64 个 patch 排在 2D 网格
        x = x.flatten(2)
        # flatten 第 2 维起：[B, 256, 64]
        x = x.transpose(1, 2)
        # 交换最后两维：[B, 64, 256]  ← 终于变成 (batch, 序列长度, embed 维度)
        return x
```

**为什么用 Conv2d 而不是先切 patch 再用 Linear？** 数学上完全等价：
- 切 patch + reshape 成 `[B, 64, 48]`，再过 `Linear(48, 256)` → 结果是 `[B, 64, 256]`
- 等价于 `Conv2d(3, 256, kernel=4, stride=4)`，因为卷积本身就是把 kernel 区域展平后做线性变换

但用 Conv2d 一行就解决，并且 GPU 上更高效，所以几乎所有实现都这么写（timm 也是）。

---

### 3.2 CLS Token

从 BERT 借来的设计。它是一个**可学习的向量**，拼到序列最前面，跟所有 patch 一起参与 attention。经过 N 层 encoder 后，CLS 位置的输出向量被认为聚合了整张图像的信息，用它来做分类。

```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
```

#### nn.Parameter() 的作用

`nn.Linear(256, 10)` 内部自动创建 `self.weight` 和 `self.bias`，它们都是 `nn.Parameter`。当调用 `optimizer.step()` 时，优化器遍历 `model.parameters()` 找到所有 Parameter 来更新。

但 CLS token 不属于任何现成的层（它不是 Linear 的权重，也不是 Conv 的核），它是**自己发明的一个可学习向量**。如果只写 `self.cls_token = torch.zeros(1, 1, 256)`，那它只是一个普通 tensor，`model.parameters()` 找不到它，优化器不会更新它，反传时梯度算了也白算。

`nn.Parameter()` 做的事就是**把一个 tensor 注册为模型的参数**，让它能被优化器发现和更新。

```python
# 普通 tensor：不会被优化器更新
self.cls_token = torch.zeros(1, 1, 256)        # ✗ 优化器看不见

# Parameter：会被优化器更新
self.cls_token = nn.Parameter(torch.zeros(1, 1, 256))  # ✓ 自动注册

# 验证：
model = VisionTransformer()
for name, param in model.named_parameters():
    print(name, param.shape)
# 输出里能看到 "cls_token torch.Size([1, 1, 256])"
```

`self.pos_embed = nn.Parameter(...)` 也是同理——位置编码也是自己发明的可学习量，不属于任何标准层。

#### 形状 [1, 1, embed_dim] 的含义

三个维度分别是：

```
[batch=1, seq_len=1, embed_dim=256]
```

**seq_len=1 是因为 CLS token 就是一个 token**。不是 64 个，不是 2 个，就是 1 个。之后要把它拼到 64 个 patch token 前面：

```python
# x: [B, 64, 256]  ← 64 个 patch token
# cls_tokens: [B, 1, 256]  ← 1 个 CLS token
x = torch.cat([cls_tokens, x], dim=1)
# 结果: [B, 65, 256]  ← 64+1=65
```

`torch.cat` 沿 dim=1（序列维）拼接，要求除了 dim=1 之外的维度必须相同。CLS 在 dim=1 上是 1，patch 在 dim=1 上是 64，拼完就是 65。

**CLS token 拼在最前面位置 0 只是习惯**，拼在末尾也行。重要的是位置必须固定（每张图都一样），这样模型才能学到"这个位置是要被读出的特殊位置"。

#### expand 广播机制

CLS token 定义时 batch 维是 1，但每个 batch 里的每张图都需要自己的 CLS。广播发生在 `expand` 这一步：

```python
# self.cls_token: [1, 1, 256]  ← 全模型只有一份
B = 8  # 假设 batch_size=8
cls_tokens = self.cls_token.expand(B, -1, -1)
# expand 后: [8, 1, 256]
# -1 表示"这个维度保持不变"
```

**`expand` 不复制内存**。8 个样本共享同一块底层数据，只是创建了一个"看起来是 [8,1,256]"的视图，节省显存。

虽然 8 个样本从同一个 CLS 出发，但经过 `cat → 加 pos_embed → 过 encoder` 之后，每个样本的 CLS 向量已经完全不同了（因为每张图的 patch 内容不同，attention 输出就不同）。反传时，8 条梯度路径回到 `self.cls_token` 这一个参数，PyTorch 自动把它们**求和**——这就是广播反传的标准行为，和 bias 的广播梯度一样。

验证代码：

```python
cls = nn.Parameter(torch.tensor([[[1.0, 2.0]]]))  # [1, 1, 2]
expanded = cls.expand(3, -1, -1)                    # [3, 1, 2]，三个样本
loss = expanded.sum()
loss.backward()
print(cls.grad)
# tensor([[[3., 3.]]])
# 每个元素的梯度=3，因为 3 个样本各贡献了 1 的梯度，求和得 3
```

整个流程：

```
定义时：cls_token [1, 1, 256]  ← 一份参数
    ↓ expand
前向时：cls_tokens [B, 1, 256]  ← B 个视图，不复制
    ↓ cat with patches
         [B, 65, 256]          ← 每个样本有自己的 CLS+patches
    ↓ encoder layers
         [B, 65, 256]          ← 每个样本的 CLS 已经不同了
    ↓ 反传
梯度回到 cls_token [1, 1, 256]  ← B 条梯度求和
```

---

### 3.3 位置编码

```python
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
# 形状 [1, 65, 256]：num_patches+1 是因为要给 CLS token 也留一个位置
```

ViT 用**可学习的位置编码**（不是 NLP Transformer 那种正弦位置编码），实验显示差别不大但更简单。

#### 为什么需要位置编码：置换不变性

需要区分两件事：**self-attention 这个运算本身**的性质，和**加了位置编码之后整个模型**的行为。

**Self-attention 本身是置换不变的。** 用最小的例子说明：假设 3 个 token，没有位置编码：

```
输入 X = [x₁, x₂, x₃]，形状 [3, d]
```

Attention 做的事是：每个 token 算 Q、K、V，然后 `softmax(QKᵀ/√d) · V`。

把输入打乱变成 `X' = [x₃, x₁, x₂]`：

```
原始：token 1 对 token 2 的注意力权重 = softmax(q₁·k₂ᵀ / √d)
打乱后：token 1（其实是原来的 x₃）对 token 2（其实是原来的 x₁）的注意力 = softmax(q₃·k₁ᵀ / √d)
```

**所有 token 之间的两两点积都还在，只是在矩阵里换了个位置**。输出也只是跟着换了位置：

```
Attention(X)  = [out₁, out₂, out₃]
Attention(X') = [out₃, out₁, out₂]   ← 输出跟着输入一起打乱了
```

这就是「置换等变」（permutation equivariant）：self-attention 完全不在乎 token 的顺序，它只关心每对 token 之间的内容相似度。它把输入当成一个**集合**（set），不是一个**序列**（sequence）。

可以想一张桌子上摆了 3 个棋子，每个棋子只看其他棋子的颜色和形状来决定自己怎么变。不管怎么重新排列棋子，只要棋子本身没变，每个棋子的变化也不会变——只不过输出的排列跟着输入走。

#### 位置编码打破置换不变性

加位置编码后：

```
原始输入：[x₁, x₂, x₃]
加位置编码后：[x₁ + pe₁, x₂ + pe₂, x₃ + pe₃]
```

如果打乱输入变成 `[x₃, x₁, x₂]`，但位置编码还是按位置分配：

```
打乱后加位置编码：[x₃ + pe₁, x₁ + pe₂, x₂ + pe₃]
```

`x₃ + pe₁` 和 `x₃ + pe₃` **不是同一个向量**了！所以现在 attention 算出来的结果会不一样——模型能区分"x₃ 在位置 1"和"x₃ 在位置 3"这两种情况。

因果关系：

```
self-attention 天生不知道顺序
    → 所以必须加位置编码
        → 加完之后整个模型才有了位置感知能力
```

如果 self-attention 本身就像 CNN 那样天然知道空间位置（卷积核的位置是固定的），就不需要位置编码了。这正是 ViT 和 CNN 的核心区别：CNN 的归纳偏置里**内置**了空间结构，ViT 要**从数据里学**（通过位置编码 + attention 模式）。

---

### 3.4 初始化：Truncated Normal

```python
nn.init.trunc_normal_(self.pos_embed, std=0.02)
nn.init.trunc_normal_(self.cls_token, std=0.02)
```

#### 普通正态初始化的问题

如果用 `torch.randn` 从标准正态 N(0, σ²) 采样，理论上有极小概率采到很大的值（比如 ±5σ）。对于一个 [65, 256] 的位置编码，有 16640 个元素，几乎一定会出现几个极端值。这些极端值会导致训练初期某些 token 的 attention 分数异常大，softmax 饱和，梯度消失或爆炸。

#### Truncated Normal 的做法

**从正态分布采样，但把超出 ±2σ 的值扔掉重采**：

```
普通正态 N(0, 0.02²)：
    值域 = (-∞, +∞)，理论上什么都可能采到

截断正态 TruncNormal(0, 0.02², a=-0.04, b=0.04)：
    值域 = [-0.04, 0.04]，超出范围的重新采
```

直觉上就是：正态分布的中间部分保留，两头的尾巴砍掉。

#### `nn.init.trunc_normal_` 的参数

```python
nn.init.trunc_normal_(
    tensor,      # 要初始化的 tensor，会被原地修改（所以函数名带下划线 _）
    mean=0.0,    # 均值，一般都是 0
    std=0.02,    # 标准差，ViT/BERT 的标准值
    a=-2.0,      # 截断下界，默认 -2（即 -2σ 处）
    b=2.0        # 截断上界，默认 +2（即 +2σ 处）
)
```

`a` 和 `b` 的默认值是 ±2.0，但这里的 2.0 是**以标准差为单位**的。当 `std=0.02` 时，实际截断范围是 `[-2×0.02, 2×0.02] = [-0.04, 0.04]`。

**为什么 std=0.02？** 来自 BERT 论文，后来 ViT 直接沿用。核心考量是：Transformer 的残差结构会让每一层的输出叠加到前面，如果初始化值太大，经过 12 层叠加后输出会爆炸。0.02 这个量级保证初始时每一层的贡献都很小，模型输出接近 0，训练从一个温和的起点开始。

#### 验证代码

```python
import torch
import torch.nn as nn

# 普通正态
a = torch.empty(100000)
nn.init.normal_(a, mean=0, std=0.02)
print(f"普通正态  min={a.min():.4f}  max={a.max():.4f}")
# 典型输出：min=-0.0892  max=0.0847  ← 偶尔出现 4σ 的极端值

# 截断正态
b = torch.empty(100000)
nn.init.trunc_normal_(b, mean=0, std=0.02, a=-2.0, b=2.0)
print(f"截断正态  min={b.min():.4f}  max={b.max():.4f}")
# 典型输出：min=-0.0400  max=0.0400  ← 严格被截断在 ±0.04 内
```

#### 在 ViT 里哪些参数用了它

```python
nn.init.trunc_normal_(self.pos_embed, std=0.02)   # 位置编码，16640 个值
nn.init.trunc_normal_(self.cls_token, std=0.02)    # CLS token，256 个值
nn.init.trunc_normal_(m.weight, std=0.02)          # 所有 Linear 层权重
```

**为什么不用 PyTorch 默认的 Kaiming 初始化？** Kaiming 初始化是为 ReLU + CNN 设计的，根据 fan_in/fan_out 自适应缩放。但 Transformer 用的是 GELU + LayerNorm + 残差，信号流的统计特性不同。实验表明 `trunc_normal_(std=0.02)` 在 Transformer 系列模型上更稳定。timm、HuggingFace 的 ViT/BERT 实现全是这个值。

---

### 3.5 Transformer Encoder Block

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        # Pre-LN：LayerNorm 放在 attention/MLP 之前，而不是之后
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=dropout,
            batch_first=True   # 用 [B, L, D] 而不是 [L, B, D]，省得 transpose
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP（FFN）部分，隐藏维度通常是 embed_dim 的 4 倍
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),                     # ViT 用 GELU，不用 ReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, 65, 256]

        # ---- 子层 1：多头自注意力 ----
        x_norm = self.norm1(x)            # 先 norm
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # Q=K=V，self-attention
        x = x + attn_out                  # 残差连接

        # ---- 子层 2：MLP ----
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x
```

#### `__init__` vs `forward` 是两个完全不同的阶段

```python
# 阶段 1：__init__（建造阶段，只执行一次）
# 告诉 PyTorch "我要造一个什么样的注意力层"
self.attn = nn.MultiheadAttention(
    embed_dim=256,      # 每个 token 的维度
    num_heads=8,        # 头数
    dropout=0.1,        # dropout 概率
    batch_first=True    # 输入格式选择
)
# 这一步创建了 Q/K/V 的投影矩阵 W_Q, W_K, W_V, W_O
# 它们的形状由 embed_dim 和 num_heads 决定
# 这些矩阵被存在 self.attn 里面，等着被用

# 阶段 2：forward（使用阶段，每次前向传播都执行）
# 告诉 PyTorch "用这些数据来算注意力"
attn_out, attn_weights = self.attn(query, key, value)
# 这里传入的是实际的数据 tensor
```

和 `nn.Linear` 一样的模式：

```python
# __init__：造一个线性层，传入的是维度参数
self.fc = nn.Linear(256, 10)       # 参数：in_features, out_features

# forward：用这个层，传入的是数据
output = self.fc(x)                # 参数：实际的 tensor [B, 256]
```

`self.attn(x_norm, x_norm, x_norm)` 传入的三个参数是 **query, key, value 三个数据 tensor**，不是配置参数。因为是 self-attention，Q=K=V 都是同一个 `x_norm`。如果是 cross-attention（比如 ACT 的 decoder 里），就会不一样：

```python
# self-attention：Q=K=V 来自同一个序列
self.attn(x, x, x)

# cross-attention：Q 来自 decoder，K=V 来自 encoder
self.attn(decoder_tokens, encoder_output, encoder_output)
```

#### `batch_first=True` 的含义

PyTorch 的 `nn.MultiheadAttention` 默认期望输入形状是 `[L, B, D]`（序列长度在前，batch 在中间），这是 PyTorch 早期的设计选择，很反直觉。

```python
# batch_first=False（默认）：
# 输入形状 [L, B, D] = [65, 8, 256]
#   L=65 序列长度在第 0 维
#   B=8  batch 在第 1 维

# batch_first=True：
# 输入形状 [B, L, D] = [8, 65, 256]
#   B=8  batch 在第 0 维
#   L=65 序列长度在第 1 维
```

整个 ViT 代码里所有 tensor 都是 `[B, L, D]` 的格式。不设 `batch_first=True` 就要在每次调用 attn 前后手动 transpose：

```python
# 不设 batch_first 的麻烦写法：
x = x.transpose(0, 1)          # [B,65,256] → [65,B,256]
attn_out, _ = self.attn(x, x, x)
attn_out = attn_out.transpose(0, 1)  # [65,B,256] → [B,65,256]

# 设了 batch_first=True 的写法：
attn_out, _ = self.attn(x, x, x)    # 直接用 [B,65,256]，搞定
```

纯粹是省事，数学上完全等价。

#### Pre-LN vs Post-LN：为什么最后还要多一个 LayerNorm

把 block 的 forward 展开，仔细看数据流的**最后一步**：

```python
def forward(self, x):
    # 子层 1
    x_norm = self.norm1(x)          # ← norm 了
    attn_out, _ = self.attn(x_norm, x_norm, x_norm)
    x = x + attn_out                # ← 残差加回来，加的是未 norm 的 x！

    # 子层 2
    x_norm = self.norm2(x)          # ← norm 了
    x = x + self.mlp(x_norm)        # ← 残差又加回来，又是未 norm 的 x！
    return x                         # ← 返回的是"norm 过的 + 未 norm 的"混合物
```

关键：**Pre-LN 是先 norm 再做变换，然后残差把未 norm 的原始值加回来了**。block 输出并不是"干净的 norm 过的"tensor。

```
Pre-LN 的数据流（ViT 用的）：

x ──────────────────┐
│                    │ 残差
├→ LN → Attn ───────┘
│                    
x' ─────────────────┐
│                    │ 残差
├→ LN → MLP ────────┘
│
x'' ← 这个输出里混着没 norm 过的成分！
→ 所以最后需要一个额外的 LN 来"收尾"
```

```
Post-LN 的数据流（原版 Transformer 用的）：

x ──────────────────┐
│                    │ 残差
├→ Attn ────────────┘
│
├→ LN                 ← norm 在残差之后
│
x' ─────────────────┐
│                    │ 残差
├→ MLP ─────────────┘
│
├→ LN                 ← norm 在残差之后
│
x'' ← 这个输出已经被 norm 过了，不需要额外 LN
```

用具体数值感受：假设最后一个 block 的 MLP 输出某个位置的值是 `[0.3, -0.1, 0.5, ...]`，而残差连接的 x 那个位置是 `[12.0, -8.0, 15.0, ...]`（经过 6 层累加，值可能比较大）：

```python
# block 的最终输出
x = x + self.mlp(x_norm)
# = [12.0, -8.0, 15.0, ...] + [0.3, -0.1, 0.5, ...]
# = [12.3, -8.1, 15.5, ...]  ← 量级主要由未 norm 的残差决定

# 如果不加最后的 LN，直接取 CLS 送分类头：
# Linear 接收的输入量级不稳定、分布不标准

# 加了最后的 LN：
x = self.norm(x)  # 重新归一化到均值 0、方差 1 附近
# Linear 接收的输入分布干净稳定
```

总结：

```
Post-LN：每个子层结束时都 norm 了 → 不需要额外 LN
Pre-LN：norm 在子层开头，残差把未 norm 的值加回来 → 最终输出没被 norm
         → 必须在所有 block 之后补一个 LN 收尾
```

去掉这个收尾 LN 训练会不稳定。timm 的 ViT、HuggingFace 的 BERT 都有。

#### GELU 激活函数

`GELU(x) ≈ x · Φ(x)`，比 ReLU 平滑，导数处处非零。直觉上 GELU 像"概率性 ReLU"——根据输入大小决定保留多少。BERT/GPT/ViT 都用它。

---

### 3.6 完整 ViT 拼装

```python
class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=4,
                 in_channels=3,
                 embed_dim=256,
                 num_heads=8,
                 num_layers=6,
                 mlp_ratio=4.0,
                 num_classes=10,
                 dropout=0.1):
        super().__init__()

        # 1) Patch embedding
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches  # 64

        # 2) CLS token 和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # 3) Encoder：堆叠 num_layers 个 block
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)   # 最后再做一次 LN（Pre-LN 架构标配）

        # 4) 分类头
        self.head = nn.Linear(embed_dim, num_classes)

        # 5) 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, 3, 32, 32]
        B = x.shape[0]

        # Step 1: patch embed → [B, 64, 256]
        x = self.patch_embed(x)

        # Step 2: 拼 CLS token
        # cls_token 形状 [1, 1, 256]，expand 到 [B, 1, 256]（不复制内存，只是视图）
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # 沿序列维拼起来 → [B, 65, 256]
        x = torch.cat([cls_tokens, x], dim=1)

        # Step 3: 加位置编码（广播：pos_embed[1,65,256] + x[B,65,256]）
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Step 4: 过 N 层 encoder
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)  # [B, 65, 256]

        # Step 5: 取出 CLS 位置的向量作为图像表征
        cls_output = x[:, 0]  # [B, 256]

        # Step 6: 分类
        logits = self.head(cls_output)  # [B, 10]
        return logits
```

验证形状：

```python
model = VisionTransformer()
x = torch.randn(8, 3, 32, 32)
out = model(x)
print(out.shape)  # torch.Size([8, 10])
print(sum(p.numel() for p in model.parameters()))  # 约 4.8M 参数
```

---

## 四、一次前向传播的具体张量流（带数字）

batch=8, 一张 32×32×3 CIFAR-10 图像走完整个模型：

| 步骤                    | 操作                                   | 形状             | 备注                   |
| --------------------- | ------------------------------------ | :------------- | -------------------- |
| 输入                    | —                                    | [8, 3, 32, 32] | 像素值                  |
| Patch 投影              | `Conv2d(3,256,k=4,s=4)`              | [8, 256, 8, 8] | 8×8=64 个 patch       |
| flatten+transpose     |                                      | [8, 64, 256]   | 变成序列                 |
| 拼 CLS                 | `cat([cls, x], dim=1)`               | [8, 65, 256]   | 多一个 token            |
| +pos_embed            | 逐元素加                                 | [8, 65, 256]   | 注入位置信息               |
| B             k 1 LN1 | LayerNorm                            | [8, 65, 256]   |                      |
| Block 1 Attn          | Q,K,V 都是 [8,65,256]，多头拆为 8 头，每头 32 维 | [8, 65, 256]   | 注意力矩阵 [8, 8, 65, 65] |
| Block 1 残差            | x + attn_out                         | [8, 65, 256]   |                      |
| Block 1 MLP           | 256→1024→256                         | [8, 65, 256]   | mlp_ratio=4          |
| Block 2~6             | 同上                                   | [8, 65, 256]   |                      |
| 最终 LN                 |                                      | [8, 65, 256]   |                      |
| 取 CLS                 | `x[:, 0]`                            | [8, 256]       | 丢掉 64 个 patch 位置     |
| 分类头                   | `Linear(256, 10)`                    | [8, 10]        | logits               |

Attention 矩阵 `softmax(QK^T / √d_k)` 那一步形状是 `[B=8, heads=8, 65, 65]`——这就是 ViT 的算力大头。**每两个 token 之间都要算一个相似度**，复杂度 `O(L²·d)`。对 32×32 + patch 4 来说 L=65 没问题，但如果换成 224×224 + patch 16，L=197 也还行；要是想做 patch 8，L 就变成 785，attention 矩阵直接 4 倍大，显存吃不消。这就是为什么后来出现 Swin Transformer 用 window attention。

---

## 五、反向传播


### 5.1 全部可训练参数

![[Backpropagation.png]]
### 5.2 梯度流动关键点

**(1) CLS token 收到所有 patch 的梯度。** 前向时 CLS 通过 attention 与所有 64 个 patch 交互；最终只有 CLS 位置的输出参与 loss。反传时，loss 的梯度从 CLS 位置流入最后一层 LN，再经过每层 attention 反向时，会同时**沿着 attention 权重的方向流回所有 patch**。也就是说 patch 位置的输出虽然不直接进 loss，它们的参数照样会被更新——通过它们对 CLS 的影响。

**(2) 残差连接是"梯度高速公路"。** 每个 `+` 节点在前向时做加法 `x = x + sublayer(LN(x))`，反传时加法的梯度**同时流向两条路**：一条走残差直通（梯度不衰减），一条走子层（经过 MLP 或 Attention 反传）。即使子层的梯度消失了，直通路径保证梯度能到达早期层。这是 Transformer 能堆 12、24 甚至 96 层而不梯度消失的核心原因。

**(3) 位置编码 pos_embed 是参数。** 每次反传，pos_embed 的 65×256 个值都会收到梯度。训练完一个 ViT 后，可视化 pos_embed[i] 和 pos_embed[j] 的余弦相似度，会发现它学到了 2D 空间结构（相邻 patch 的 pos_embed 相似度高）——这是 ViT 论文里一个经典图。B 个样本的梯度求和到这一个参数上。

**(4) cls_token 的梯度。** B 个样本从同一个 cls_token expand 出来，反传时 B 条梯度求和回到这一个参数（和 bias 的广播梯度一样）。

**(5) Patch embedding 的梯度回到像素。** `Conv2d(3,256,k=4,s=4)` 的权重形状是 `[256, 3, 4, 4]`，会收到梯度。因为 stride=patch_size，每个像素只贡献给一个 patch（非重叠），反传非常简单。

#### 为什么 stride=patch_size 时反传简单

普通卷积 `stride=1, kernel=3` 时，相邻窗口大量重叠：

```
图像一行像素：  [p1] [p2] [p3] [p4] [p5] [p6] [p7] [p8]

stride=1, kernel=3:
窗口1: [p1  p2  p3]
窗口2:     [p2  p3  p4]
窗口3:         [p3  p4  p5]
...

p3 被窗口 1、2、3 都用到了！
反传时：grad_p3 = grad_out1 · w[2] + grad_out2 · w[1] + grad_out3 · w[0]
需要累加多个梯度贡献
```

ViT 的 Patch Embedding `stride=4, kernel=4`，不重叠：

```
stride=4, kernel=4:
patch 1: [p1  p2  p3  p4]
patch 2:                   [p5  p6  p7  p8]

每个像素只属于一个 patch，没有重叠！
反传时：grad_p3 = grad_patch1 · w[2]
只有一条梯度路径，没有累加
```

2D 情况完全一样，32×32 图像被切成 8×8=64 个不重叠的 4×4 方块，没有任何像素同时属于两个 patch。

**(6) 梯度爆炸的风险**：原版 ViT 在大模型时偶尔训练不稳定，常见做法是梯度裁剪 `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`。

---

## 六、训练流程

### 6.1 数据增强（逐步详解）

```python
train_transform = transforms.Compose([   # Compose：把多个变换串成流水线，按顺序执行
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])
```

#### RandomCrop(32, padding=4)

先在图像四周**填充 4 个像素**（默认填 0，即黑色），图像从 32×32 变成 40×40。然后从 40×40 中**随机裁出一个 32×32 的区域**。

```
原图 32×32:          填充后 40×40:              随机裁剪 32×32:
┌──────────┐      ┌──────────────────┐      ┌──────────────────┐
│          │      │ 0000000000000000 │      │                  │
│  原始图  │  →   │ 00 ┌────────┐ 00│  →   │   ┌──────────┐   │
│          │      │ 00 │ 原始图 │ 00│      │   │随机选这块│   │
│          │      │ 00 └────────┘ 00│      │   └──────────┘   │
└──────────┘      │ 0000000000000000 │      └──────────────────┘
                  └──────────────────┘
```

**效果**：图像整体随机平移最多 ±4 像素。模型被迫学习"猫在哪里都是猫"，即平移不变性。

参数：`32` = 输出尺寸（裁完还是 32×32），`padding=4` = 四边各填充 4 像素。

#### RandomHorizontalFlip()

以 **50% 概率**水平翻转图像（左右镜像）。不传参数时默认 `p=0.5`。

让模型知道"向左的猫"和"向右的猫"是同一类。不做垂直翻转，因为现实中物体很少上下颠倒。

#### RandAugment(num_ops=2, magnitude=9)

**自动数据增强策略**。内部有一个变换池，包含 14 种操作：旋转、色彩偏移、对比度调整、锐化、亮度变化、裁切、等化直方图等等。

每次调用时，从池中**随机选 `num_ops` 个操作**，每个操作的**强度统一设为 `magnitude`**（范围 0~30，越大越激烈）。

```
变换池: [旋转, 色彩, 对比度, 锐化, 亮度, 剪切, 平移, 均衡化, ...]
                                    ↓ 随机选 2 个
本次选中: [对比度(强度9), 旋转(强度9)]
下一张图可能选: [亮度(强度9), 剪切(强度9)]
```

参数：`num_ops=2` = 每张图随机选 2 种变换，`magnitude=9` = 变换强度（9 是中等偏强）。

**ViT 特别需要这个**：CNN 内置局部性和平移不变性偏置，对数据量要求低。ViT 什么偏置都没有，全靠数据学，必须用强增强来"人工扩大"数据集多样性，否则在小数据集上严重过拟合。DeiT 论文加了强增强+蒸馏才把 ViT 在 ImageNet-1K 上追平 CNN。

#### ToTensor()

做两件事：
1. 把 PIL Image（numpy 数组）转成 `torch.Tensor`
2. 把像素值从 `[0, 255]` 整数归一化到 `[0.0, 1.0]` 浮点数

```
输入：PIL Image, shape (H, W, C), dtype=uint8, 值域 [0, 255]
输出：torch.Tensor, shape (C, H, W), dtype=float32, 值域 [0.0, 1.0]
```

还把维度顺序从 `(H, W, C)` 变成了 `(C, H, W)`——PyTorch 的 Conv2d 要求通道在前面。

#### Normalize(mean, std)

对每个通道做标准化：`output = (input - mean) / std`

```
R 通道：(pixel - 0.4914) / 0.2470
G 通道：(pixel - 0.4822) / 0.2435
B 通道：(pixel - 0.4465) / 0.2616
```

**这些数字哪来的？** 是 CIFAR-10 训练集 50000 张图所有像素的统计值。比如 R 通道的均值 0.4914 意思是 CIFAR-10 所有图的红色通道 ToTensor 后的平均值是 0.4914。每个数据集有自己的值（ImageNet 的又不同）。

**效果**：归一化后每个通道的分布近似 N(0, 1)，让模型训练更稳定。如果不做这步，大值通道会主导梯度更新方向。

#### 测试集 transform

```python
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])
```

只有 ToTensor + Normalize，**没有任何随机增强**。测试时要评估模型在"真实"图像上的性能，随机增强会导致每次测出的准确率不同，无法公平比较。Normalize 必须和训练时用**完全相同的 mean/std**，否则输入分布不匹配，模型表现会崩。

#### 整个流水线的执行顺序

每取一张训练图像时：

```
磁盘上的 CIFAR-10 图 (PIL Image, 32×32)
    ↓ RandomCrop(32, padding=4)    ← 填充→裁剪，随机平移
    ↓ RandomHorizontalFlip()       ← 50%概率左右翻
    ↓ RandAugment(2, 9)           ← 随机选 2 种强变换
    ↓ ToTensor()                   ← 转 tensor, [0,255]→[0,1]
    ↓ Normalize(mean, std)         ← 标准化到 ~N(0,1)
最终输出: tensor [3, 32, 32], 值域大约 [-2, +2]
```

同一张图每个 epoch 经过这个流水线都会产生不同的结果（因为前三步是随机的），相当于模型每次看到的都是"略有不同的版本"。

### 6.2 损失函数

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

#### CrossEntropyLoss 本身

分类任务的标准损失。内部做两步：先对 logits 做 softmax 变成概率，再算交叉熵。

```
logits = [2.1, 0.3, -1.5, ...]   ← 模型输出，10 个数（CIFAR-10 的 10 类）
                ↓ softmax
probs  = [0.72, 0.12, 0.02, ...]  ← 变成概率，总和=1
                ↓ 和真实标签算交叉熵
loss   = -log(probs[正确类别])
```

如果正确类别是第 0 类，`loss = -log(0.72) = 0.33`。模型越确信正确答案，概率越接近 1，loss 越接近 0。

#### label_smoothing=0.1 的作用

不加 label_smoothing 时，目标是 one-hot：

```
正确类别=3，10 分类
标准 one-hot:  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                              ↑ 要求模型对正确类 100% 确信
```

模型被逼着把正确类的概率推到 1.0，其他类推到 0.0。这要求 logits 差距非常大（softmax 要输出接近 1 需要正确类的 logit 远大于其他类）。后果是模型变得**过度自信**，权重被推向极端值，泛化能力变差。

加了 `label_smoothing=0.1` 后，目标分布变成：

```
ε = 0.1, K = 10（类别数）

正确类的目标：1 - ε + ε/K = 1 - 0.1 + 0.01 = 0.91
其他类的目标：ε/K = 0.1/10 = 0.01

平滑后:  [0.01, 0.01, 0.01, 0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
```

模型不需要把正确类推到 1.0，推到 0.91 就够了。让模型"谦虚"一点，权重不会走极端，泛化更好。

### 6.3 优化器

```python
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
```

#### `model.parameters()`

告诉优化器要更新哪些参数。返回模型里所有 `nn.Parameter` 的迭代器，包括 Conv2d 权重、cls_token、pos_embed、所有 Linear 的权重和偏置、LayerNorm 的 γ 和 β——就是反传流程图里所有可训练参数。

#### `lr=3e-4`（学习率）

每次参数更新的步长。3e-4 = 0.0003。

```
参数更新公式（简化版）：
θ_new = θ_old - lr × gradient
      = θ_old - 0.0003 × gradient
```

太大（比如 0.01）：参数跳来跳去，loss 不收敛。太小（比如 0.00001）：训练极慢，可能卡在局部最优。3e-4 是 Adam 系优化器在 Transformer 上经验验证的好起点。

#### `weight_decay=0.05`（权重衰减）

每次更新时把参数往 0 的方向拉一点，防止权重变太大。

```
AdamW 的更新（简化）：
θ_new = θ_old - lr × (adam_gradient + weight_decay × θ_old)
      = θ_old - 0.0003 × (adam_gradient + 0.05 × θ_old)
```

`0.05 × θ_old` 这一项让大参数受到更大的惩罚——参数越大，被拉回 0 的力越强。效果就是正则化，抑制过拟合。

**为什么 ViT 用 0.05 这么大？** CNN 一般用 1e-4，ViT 用 0.05 是 CNN 的 500 倍。因为 ViT 没有归纳偏置，参数自由度极高，在小数据集上容易过拟合，需要更强的正则化。

#### AdamW vs 普通 Adam

区别在于 weight decay 的实现方式。普通 Adam 把 weight decay 加在梯度里（L2 正则），但 Adam 的自适应学习率会缩放这个正则项，导致正则效果被削弱。AdamW 把 weight decay 从梯度计算中**解耦**出来，直接在参数上操作：

```
Adam:   gradient' = gradient + wd × θ     ← weight decay 混在梯度里
        θ = θ - lr × Adam(gradient')       ← Adam 的自适应缩放会影响 wd 的效果

AdamW:  θ = θ - lr × Adam(gradient)        ← Adam 只处理纯梯度
        θ = θ - lr × wd × θ               ← weight decay 单独作用，不被缩放
```

结果是 AdamW 的正则化更稳定可控。所有现代 Transformer 都用 AdamW。

### 6.4 学习率调度器

```python
num_epochs = 200
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
```

#### 为什么需要调度器

训练初期 loss 高，需要大步长快速下降。训练后期接近最优点，大步长会在最优点附近反复震荡，需要小步长精细调整。所以学习率应该**随训练逐渐降低**。

#### CosineAnnealingLR 的公式

```
lr_t = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T_max))
```

默认 `lr_min=0`（也叫 `eta_min`），所以：

```
lr_t = 0.5 × lr_max × (1 + cos(π × t / T_max))
```

#### `T_max=200`

整个余弦周期的长度，设成和总 epoch 数相同，学习率从头到尾走完一个半余弦：

```
lr
0.0003 |*
       | **
       |   **
       |     ***
       |        ***
       |           ****
       |               *****
       |                    ********
0      |                            **********
       └──────────────────────────────────────→ epoch
       0          50        100       150   200
```

具体数值：

```
epoch=0:   lr = 0.5 × 3e-4 × (1 + cos(0))       = 3e-4    ← 最大
epoch=50:  lr = 0.5 × 3e-4 × (1 + cos(π/4))      ≈ 2.6e-4
epoch=100: lr = 0.5 × 3e-4 × (1 + cos(π/2))      = 1.5e-4  ← 半衰
epoch=150: lr = 0.5 × 3e-4 × (1 + cos(3π/4))     ≈ 4.4e-5
epoch=200: lr = 0.5 × 3e-4 × (1 + cos(π))        = 0       ← 归零
```

#### 为什么用余弦而不是线性衰减

余弦衰减在前半段下降慢（给模型足够时间探索），后半段下降快（加速收敛到最优点）。实验表明这个形状在 Transformer 上比线性衰减或阶梯衰减效果更好。ViT 原论文和 DeiT 都用余弦。

### 6.5 训练循环

```python
for epoch in range(num_epochs):
    model.train()                   # 开启 Dropout
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        logits = model(imgs)              # 前向
        loss = criterion(logits, labels)  # CE loss

        optimizer.zero_grad()
        loss.backward()                   # 反向
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()                  # 更新参数

    scheduler.step()                # 每 epoch 更新学习率

    # ---- 测试 ----
    model.eval()                    # 关闭 Dropout
    correct, total = 0, 0
    with torch.no_grad():           # 关闭梯度追踪，省显存
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Epoch {epoch}: test acc = {correct/total:.4f}")
```

#### 训练 vs 测试的区别

- `model.train()` vs `model.eval()` 切换 Dropout 和 LayerNorm 行为（LN 其实在 train/eval 行为一样，只有 BN 有区别，但习惯还是要切换）
- `with torch.no_grad()` 关闭梯度追踪，省显存
- 测试时不做数据增强，只做归一化

---

## 七、与机器人操控研究的联系

- **ViT 和 ACT 的视觉编码器**：ACT 默认用 ResNet18 当视觉骨干，但可以替换成 ViT。区别在于 ResNet 输出空间特征图 `[B, C, H, W]`，ACT 把它 flatten 当 token 序列；ViT 本身就输出 token 序列 `[B, L, D]`，更自然。Phantom 和很多新一代 manipulation 论文已经在用 ViT 作为视觉骨干。

- **和 ATM/Point Policy 的关系**：ATM 用 CoTracker 提点的轨迹，每个点其实就是一个 spatio-temporal token。可以把它和 ViT 的 patch token 拼起来一起喂给 Transformer——这种"多源 token"的设计非常符合 Route A 思路。

- **ACT 里的 transformer encoder 和 ViT 几乎一样**：ACT 本质就是把视觉特征 + 关节状态当成 token，送给 encoder-decoder。ViT 给的是"视觉怎么变成 token"的标准答案，剩下的全是 Transformer。
