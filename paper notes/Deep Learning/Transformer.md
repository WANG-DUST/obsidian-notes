---
tags:
  - transformer
  - attention
  - deep-learning
  - pytorch
date: '2025-06-16'
status: 完成
ref: 'https://zh-v2.d2l.ai/chapter_attention-mechanisms/transformer.html'
---
# Transformer 架构逐层拆解

![[Construct.png|597]]

---

## 0. 全局概览：数据流

![[global_flow.png|597]]

```
源序列 "I love you"
  ↓
[嵌入层] → (batch, seq_len, d_model)
  ↓
[+ 位置编码] → (batch, seq_len, d_model)
  ↓
[编码器 × N] → enc_output: (batch, src_len, d_model)
  ↓                                ↓ (作为 K, V 送入解码器)
  ↓                          目标序列 "<bos> 我 爱 你"
  ↓                                ↓
  ↓                          [嵌入层] → (batch, tgt_len, d_model)
  ↓                                ↓
  ↓                          [+ 位置编码]
  ↓                                ↓
  ↓                          [解码器 × N]
  ↓                                ↓
  ↓                          [全连接层] → (batch, tgt_len, vocab_size)
  ↓                                ↓
  ↓                          预测: "我 爱 你 <eos>"
```

用一个具体例子贯穿全文：
- `d_model = 512`（隐藏维度）
- `n_heads = 8`（注意力头数）
- `d_ff = 2048`（前馈网络中间维度，ff = feed-forward）
- `num_layers = 6`（编码器/解码器各 6 层）
- `vocab_size = 10000`

### forward 执行顺序对照

```python
def forward(self, src, tgt, src_mask=None, tgt_mask=None):
    # ========== 编码器 ==========
    x = self.src_embed(src)         # §1 嵌入层
    x = self.src_pos(x)             # §2 位置编码（内部完成加法 x = x + pe）
    for layer in self.encoder_layers:
        x = layer(x, src_mask)      # §8 编码器块（内含 §4自注意力 → §6AddNorm → §7FFN → §6AddNorm）
    enc_output = x

    # ========== 解码器 ==========
    x = self.tgt_embed(tgt)         # §1 嵌入层
    x = self.tgt_pos(x)             # §2 位置编码
    for layer in self.decoder_layers:
        x = layer(x, enc_output, tgt_mask, src_mask)
        # §9 解码器块（§5掩蔽自注意力 → §6 → §4交叉注意力 → §6 → §7FFN → §6）
    logits = self.output_linear(x)  # §10 全连接层
    return logits
```

---

## 1. 嵌入层（Embedding）

![[embedding_lookup.png]]
### 作用
把离散的词元 ID（整数）映射成连续的稠密向量。比如词 "love" 的 ID 是 42，嵌入层把 42 变成一个 512 维的向量。

### 输入 / 输出
- **输入**: `(batch_size, seq_len)` — 整数张量，每个元素是词表中的索引
- **输出**: `(batch_size, seq_len, d_model)` — 浮点张量

### 为什么需要它
神经网络无法直接处理离散 ID。ID 是类别标签（像球衣号码），数值大小没有意义——"transformer"=9527 不代表它比 "猫"=3 "大"。矩阵乘法 `Wx` 会把数值大小当真，结果毫无意义。此外离散整数无法求梯度——你不能对 ID=3 做微小扰动 3.001 来观察 loss 变化。

Embedding 把 ID 变成可学习的连续向量，让矩阵运算有意义、让语义相似性可以表达（点积衡量相似度：`a·b = |a|×|b|×cos(θ)`，cos(90°)=0 表示正交无关；one-hot 里任意两个不同词都正交，"猫和狗"的相似度 = "猫和引力波"的相似度 = 0）、让梯度可以流动。

### nn.Embedding 的工作原理

`nn.Embedding(vocab_size, d_model)` 内部维护一个 `(vocab_size, d_model)` 的权重矩阵。前向传播 `emb(x)` 不是矩阵乘法，而是**查表**：对输入 `(batch, seq_len)` 中的每个整数 ID，取权重矩阵对应的那一行。输入每个位置从一个标量（ID）变成一个 d_model 维向量，所以输出多了一维。

PyTorch 的 `nn.Embedding` 默认从 **N(0, 1)**（标准正态分布）初始化。

### 查表过程详解

`nn.Embedding(vocab_size=5, d_model=3)` 内部创建一个权重矩阵（查找表）：

```
emb.weight 的形状: (vocab_size, d_model) = (5, 3)

        d0     d1     d2
词 0: [ 0.12, -0.34,  0.56]
词 1: [ 0.78,  0.91, -0.23]
词 2: [-0.45,  0.67,  0.11]
词 3: [ 0.33, -0.88,  0.44]   ← "猫" 的向量
词 4: [ 0.55,  0.22, -0.77]   ← "狗" 的向量
```

这个矩阵始终是 `(vocab_size, d_model)`，它是**参数**，不是输出。前向传播 `emb(x)` 对 x 中的每个整数 ID，去权重矩阵里取出对应那一行：

```python
x = torch.tensor([[3, 4, 1]])   # (batch=1, seq_len=3)
output = emb(x)                  # (batch=1, seq_len=3, d_model=3)
```

```
ID=3 → 取第3行 → [ 0.33, -0.88,  0.44]
ID=4 → 取第4行 → [ 0.55,  0.22, -0.77]
ID=1 → 取第1行 → [ 0.78,  0.91, -0.23]

output = [[[ 0.33, -0.88,  0.44],    ← 位置0，对应ID=3
            [ 0.55,  0.22, -0.77],    ← 位置1，对应ID=4
            [ 0.78,  0.91, -0.23]]]   ← 位置2，对应ID=1

形状: (1, 3, 3) 即 (batch, seq_len, d_model)
```

注意区分**形状**和**值**：`x = [[3, 4, 1]]` 的形状是 `(1, 3)` 即 batch=1, seq_len=3。3, 4, 1 是词的 ID 编号（值），不是维度大小。

查表操作在数学上等价于 one-hot 矩阵乘法（`one_hot × weight = 对应行`），但 PyTorch 直接用整数索引取行，效果一样但快得多。

### 为什么乘以 sqrt(d_model)

位置编码用 sin/cos，值域固定在 [-1, 1]，标准差约 0.5。如果不缩放，嵌入向量的分量量级可能和位置编码差距很大，位置信号会压过词义信号。乘以 `sqrt(d_model)` 后两者在同一量级，模型既知道"这是什么词"，也知道"它在第几个位置"。

### 代码

```python
import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x: (batch, seq_len)，每个元素是词的 ID
        # self.embedding(x): 查表，输出 (batch, seq_len, d_model)
        # 乘以 sqrt(d_model) 让嵌入值和位置编码在同一数量级
        return self.embedding(x) * math.sqrt(self.d_model)
```

---

## 2. 位置编码（Positional Encoding）

![[positional_encoding.png]]

### 公式

$$PE_{(pos,\, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos,\, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中 $pos$ 是词在序列中的位置，$i$ 是维度索引，$d_{model}$ 是模型的隐藏维度。

### 作用
给每个位置注入"我是第几个词"的信息。因为自注意力是**置换不变的**（把词打乱顺序，注意力计算结果一样），所以必须额外告诉模型词的顺序。RNN 天然有顺序，但 Transformer 并行处理所有位置，没有位置编码就分不清 "狗咬人" 和 "人咬狗"。

### 嵌入和位置编码在哪里相加

封装在 `PositionalEncoding.forward()` 内部的这一行：`x = x + self.pe[:, :x.size(1), :]`。调用链是 `x = self.src_embed(src)` → `x = self.src_pos(x)`，第二步内部完成了加法。

### 代码

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维用 cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # 不参与梯度更新

    def forward(self, x):
        # x: (batch, seq_len, d_model) — 嵌入层的输出
        x = x + self.pe[:, :x.size(1), :]  # ← 加法在这里
        return self.dropout(x)
```

### 代码解析

**初始化**：`pe = torch.zeros(max_len, d_model)` 创建全零矩阵，第 i 行存放位置 i 的编码。`position` 通过 `unsqueeze(1)` 变成列向量 `(max_len, 1)`，以便与频率向量 `(d_model/2,)` 广播相乘，一次算出整个矩阵。

**div_term 频率项**：论文要求 $1/10000^{2i/d_{model}}$，代码用等价变换 `exp(2i × (-ln10000 / d_model))` 在对数空间计算，避免大数幂运算的精度问题。结果是一组递减的频率值——低维变化快捕捉近距离关系，高维变化慢捕捉远距离关系。

**sin/cos 交错填充**：`position * div_term` 广播得到 `(max_len, d_model/2)` 矩阵。`0::2` 选偶数列填 sin，`1::2` 选奇数列填 cos。虽然按列切片，但 `:` 选了所有行且每行 position 不同，所以每行仍是不同位置的编码向量。按列区分只是为了将同一频率的 sin/cos 交错排列到相邻维度，使任意两个位置的编码差可用线性变换表示。

**unsqueeze(0)**：将 `(max_len, d_model)` 变为 `(1, max_len, d_model)`，与三维输入 `(batch, seq_len, d_model)` 相加时广播机制自动扩展 batch 维，每个样本加相同编码。

**register_buffer**：将 pe 登记到 `_buffers` 表，使其跟随 `.to(device)` 移动和模型保存/加载，但不参与梯度计算。普通属性 `self.pe = pe` 不会被 `.to()` 发现（因为 `nn.Module` 只遍历 `_parameters` 和 `_buffers`）；`nn.Parameter(pe)` 会被优化器当可训练参数更新——都不合适。第一个参数 `'pe'`（字符串）是名字，第二个参数 `pe` 是张量。

**forward**：`x.size(1)` 取实际序列长度 seq_len，用切片只取前 seq_len 行编码。相加后返回 `dropout(x)` 进行正则化，防止模型过度依赖某些位置特征。

---

## 3. 缩放点积注意力（Scaled Dot-Product Attention）

![[scaled_dot_product_attention.png]]

### 作用
这是所有注意力机制的核心计算单元。给定 Q（查询）、K（键）、V（值），计算每个查询应该关注哪些键，然后对值做加权求和。

### 与多头注意力的关系

缩放点积注意力不是和多头注意力并列的第四种注意力，而是所有多头注意力**内部**共用的计算核心：

```
多头注意力（外壳）
  ├── W_q, W_k, W_v 投影
  ├── 拆分成 8 个头
  ├── 每个头调用 → 缩放点积注意力（核心计算）
  ├── 拼接 8 个头的输出
  └── W_o 投影
```
编码器的多头自注意力、解码器的掩蔽多头自注意力、解码器的交叉注意力——三者的**内部**都在调用同一个缩放点积注意力函数，区别只是传进去的 Q/K/V 来源和 mask 不同。

### 各维度参数含义

- `batch`: 同时处理几个句子，各算各的
- `n_heads`: 几个注意力头，各算各的
- `seq_q`: Query 序列长度（几个"提问者"）
- `seq_k`: Key/Value 序列长度（几个"被查者"）
- `d_k`: Query/Key 的向量维度，**唯一参与点积计算的维度**（被matmul"消掉"）
- `d_v`: Value 的向量维度，决定输出维度

Q 和 K 做点积时，`matmul` 只对最后两维做矩阵乘法，batch 和 n_heads 都是独立并行的。

### Q·K^T 点积过程详解

点积**只发生在 d_k 这一个维度上**，其他维度都是“各算各的”：

```python
scores = torch.matmul(Q, K.transpose(-2, -1))
# Q:              (batch, n_heads, seq_q, d_k)
# K.transpose:    (batch, n_heads, d_k, seq_k)
#                  ↑     ↑
#                  各算各的  各算各的
# 最后两维: (seq_q, d_k) × (d_k, seq_k) → (seq_q, seq_k)
```

忽略 batch 和 n_heads，只看一个头里面：

```
假设 seq_q=3（"我 爱 你"），seq_k=3（"I love you"），d_k=4

Q: (3, 4)                     K^T: (4, 3)
                                       I    love   you
"我" → [0.2, 0.8, 0.1, 0.5]    d0  [0.9,  0.1,  0.3]
"爱" → [0.1, 0.9, 0.3, 0.7]    d1  [0.2,  0.7,  0.8]
"你" → [0.6, 0.1, 0.4, 0.2]    d2  [0.4,  0.3,  0.1]
                                d3  [0.5,  0.6,  0.9]

scores[0][1] = "我" 和 "love" 的点积
             = 0.2×0.1 + 0.8×0.7 + 0.1×0.3 + 0.5×0.6
             = 0.02 + 0.56 + 0.03 + 0.30 = 0.91

完整 scores 矩阵:
            I     love    you
   "我"  [ 0.72,  0.91,  0.63 ]
   "爱"  [ 0.66,  1.13,  0.84 ]
   "你"  [ 0.72,  0.37,  0.44 ]
```

`scores[i][j]` = 第 i 个 Query 和第 j 个 Key 沿 d_k 维度做点积 = “第 i 个提问者和第 j 个被查者有多匹配”。“爱”对“love”的分数最高（1.13），符合直觉。

### 为什么要除以 sqrt(d_k)
Q 和 K 做点积后，结果的方差正比于 `d_k`。如果 `d_k=64`，点积值可能很大，导致 softmax 进入饱和区（梯度接近 0）。除以 `sqrt(64)=8` 让方差回到 1。

### masked_fill 怎么工作

`scores.masked_fill(mask == 0, float('-inf'))`：哪里 `mask == 0` 为 True，就把那个位置替换成 `-inf`，其他位置不动。用具体数字走一遍（解码器自注意力，目标句 "我 爱 你"）：

```
原始 scores:
          我    爱    你
  "我" [ 0.72, 0.91, 0.63 ]
  "爱" [ 0.66, 1.13, 0.84 ]
  "你" [ 0.72, 0.37, 0.44 ]

因果 mask（下三角=1，上三角=0）:
  "我" [  1,    0,    0  ]
  "爱" [  1,    1,    0  ]
  "你" [  1,    1,    1  ]

mask==0 的位置填 -inf 后:
  "我" [ 0.72, -inf,  -inf  ]
  "爱" [ 0.66,  1.13, -inf  ]
  "你" [ 0.72,  0.37,  0.44 ]
```

为什么填 -inf 而不是 0？因为下一步是 softmax，`e^(-inf) = 0`，所以 -inf 经过 softmax 变成精确的 0。以第一行为例：

```
softmax([0.72, -inf, -inf]) = [1.00, 0.00, 0.00]  ← "我"只关注自己

如果填 0:
softmax([0.72, 0, 0]) = [0.51, 0.25, 0.25]  ← "我"仍能看到"爱"和"你"，mask 失效
```

完整 softmax 结果：

```
"我" softmax[ 0.72, -inf,  -inf ] = [1.00, 0.00, 0.00]
"爱" softmax[ 0.66,  1.13, -inf ] = [0.38, 0.62, 0.00]
"你" softmax[ 0.72,  0.37,  0.44] = [0.40, 0.28, 0.32]
```

"爱"在生成时只能参考"我"和"爱"自己，完全看不到未来的"你"。

**直觉**：想象你在图书馆找书。Q 是你的需求（"机器人学"），K 是每本书的标签，V 是书的内容。注意力分数就是你的需求和每本书标签的匹配度，最终你按匹配度加权"读取"各本书的内容。

### softmax 之后得到的是"该看谁"，加权求和才是"看到了什么"

注意力到这一步的时候，已经算出了每个 Query 对每个 Key 的权重：

```
"爱" 对 [我, 爱, 你] 的注意力权重: [0.38, 0.62, 0.00]
```

这只是一组比例数字，告诉你"爱"应该 38% 关注"我"、62% 关注自己、0% 关注"你"。但**光知道该看谁没用，还得把看到的内容拿出来**。

Value 就是"内容"：

```
V_我 = [0.1, 0.9, 0.3, ...]    ← "我" 携带的语义内容
V_爱 = [0.7, 0.2, 0.8, ...]    ← "爱" 携带的语义内容
V_你 = [0.4, 0.6, 0.1, ...]    ← "你" 携带的语义内容
```

加权求和就是按权重把这些内容混合起来：

```
"爱" 的输出 = 0.38 × V_我 + 0.62 × V_爱 + 0.00 × V_你

= 0.38 × [0.1, 0.9, 0.3, ...]
+ 0.62 × [0.7, 0.2, 0.8, ...]
+ 0.00 × [0.4, 0.6, 0.1, ...]

= [0.472, 0.466, 0.610, ...]
```

这个输出向量不再是"爱"自己的孤立语义，而是一个**融合了上下文的新向量**——62% 来自自己、38% 来自"我"。

### 为什么不直接用 K 的内容而要单独搞一个 V

Q 和 K 的作用是**算匹配度**，V 的作用是**提供内容**。这两件事对向量的要求不同：

```
K 需要的是: 容易被正确的 Q 匹配上（像标签）
V 需要的是: 包含对下游有用的信息（像内容）

好标签 ≠ 好内容
```

用同一个向量同时做这两件事会互相干扰。分开之后，W_k 专门学习"怎么让匹配更准"，W_v 专门学习"传递什么信息最有用"，各自优化互不牵扯。

### 代码

```python
def scaled_dot_product_attention(Q, K, V, mask=None): 
   """ 
   Q: (batch, n_heads, seq_q, d_k) 
   K: (batch, n_heads, seq_k, d_k) 
   V: (batch, n_heads, seq_k, d_v) 
   mask: (batch, 1, 1, seq_k) 或 (batch, 1, seq_q, seq_k) 
   """ 
   d_k = Q.size(-1) # 取 Q 最后一维的大小，即 d_k 
   # 1. Q 乘以 K 的转置：(batch, n_heads, seq_q, d_k) × (batch, n_heads,             d_k, seq_k) 
   # → (batch, n_heads, seq_q, seq_k) 
   # scores[i][j] 表示第 i 个查询对第 j 个键的"原始注意力分数" 
   scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k) 
   
   # 2. 应用 mask：把不该看的位置设为 -inf 
   # softmax(-inf) = 0，这些位置的注意力权重就是 0 
   if mask is not None: 
      scores = scores.masked_fill(mask == 0, float('-inf')) 
      
   # 3. softmax 归一化：沿最后一个维度（seq_k 维），使每行权重之和为 1
   attn_weights = torch.softmax(scores, dim=-1) 
    
   # 4. 加权求和：(batch, n_heads, seq_q, seq_k) × (batch, n_heads, seq_k,           d_v) 
   # → (batch, n_heads, seq_q, d_v) 
   # 每个查询得到一个 d_v 维的输出向量 
   output = torch.matmul(attn_weights, V) 
   return output, attn_weights
```

---

## 4. 多头注意力（Multi-Head Attention）

![[multihead_attention.png]]

### 作用

把 d_model 维的 Q、K、V **拆分成 h 个头**，每个头独立做注意力计算，最后拼接起来。不同的头可以关注不同的模式（比如头1关注语法关系，头2关注语义相似性）。

### 输入 / 输出

- **输入**: Q, K, V 各 `(batch, seq_len, d_model)`，可选 mask
- **输出**: `(batch, seq_len, d_model)` — 形状不变

### 为什么用多头而不是单头

单个注意力只能学一种"关注模式"。多头让模型同时从多个角度关注信息。就像一个人同时用"语法视角"、"语义视角"、"位置视角"来理解句子。

### W_q、W_k、W_v 是什么

三个不同的线性变换，把同一个输入变成三种不同"角色"的向量。`Linear(512, 512)` 维度相同但**方向变了**——同一个输入经过不同的 W 得到完全不同的向量。如果不做投影直接 Q=K=V=x，每个词和自己的点积永远最大，模型倾向于只关注自己。

### 为什么是 Linear(512, 512) 而不是 Linear(512, 64)

实现技巧：一个 `Linear(512, 512)` 等价于 8 个 `Linear(512, 64)` 横向拼接。一次矩阵乘法算完所有头，然后 reshape 拆分——GPU 并行效率更高。

### 多头并行计算怎么实现

没有循环。拆分后 Q 形状是 `(batch, n_heads, seq, d_k)`，`matmul` 把 batch 和 n_heads 都当批次维度自动并行。

### 每个头怎么学到关注不同东西

人为设计的是**结构**（每个头有独立的投影参数），学出来的是**功能**（每个头具体关注什么）。

**随机初始化 + 梯度下降**是关键机制。初始化时 8 个头的参数从随机分布中独立采样，起点就不一样。训练时梯度反传给每个头的信号也不同：

```
头0 碰巧初始化后对“相邻词”的注意力分数略高
  → 这个方向降低了 loss → 梯度强化了这个倾向
  → 越训练越擅长捕捉局部关系

头3 碰巧初始化后对“主语-谓语”的注意力分数略高
  → 这个方向也降低了 loss → 梯度强化了这个倾向
  → 越训练越擅长捕捉语法结构
```

这和“对称性破缺”很像——8 个头一开始几乎一样，但微小的随机差异被梯度放大，最终分化出不同的功能。如果所有头初始化完全相同，梯度也完全相同，更新后还是相同——8 个头永远一样，等于只有 1 个头。

原版 Transformer 完全靠自主学习。后续有研究提出辅助损失（惩罚头之间相似性）和头剪枝（删掉无用头），但这些不是 Transformer 本身的设计。

### query、key、value 参数从哪来

| 注意力类型 | query | key | value | 三者相同？ |
|-----------|-------|-----|-------|-----------|
| 编码器自注意力 | x | x | x | 全部相同 |
| 解码器掩蔽自注意力 | x | x | x | 全部相同 |
| 解码器交叉注意力 | x（解码器） | enc_output | enc_output | Q ≠ K=V |

### 代码

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        # 确保 d_model 能被 n_heads 整除
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads   # 每个头的维度，例如 512/8=64
        self.n_heads = n_heads

        # 4 个线性投影层（W_Q, W_K, W_V, W_O）
        # 虽然有多个头，但用一个大矩阵一次性投影，然后 reshape 拆分
        self.W_q = nn.Linear(d_model, d_model)  # (512, 512)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # 输出投影

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # ============ 第1步：线性投影 ============
        # query: (batch, seq_q, d_model) → (batch, seq_q, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # ============ 第2步：拆分多头 ============
        # (batch, seq, d_model) → (batch, seq, n_heads, d_k) → (batch, n_heads, seq, d_k)
        # view: 把最后一维 d_model 拆成 (n_heads, d_k)
        # transpose(1,2): 把 n_heads 维提到 seq 前面，方便后面并行计算
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # ============ 第3步：缩放点积注意力 ============
        # 对每个头独立计算注意力（因为 n_heads 在 batch 维度，PyTorch 自动并行）
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # ============ 第4步：拼接多头 ============
        # (batch, n_heads, seq_q, d_k) → (batch, seq_q, n_heads, d_k) → (batch, seq_q, d_model)
        # contiguous(): transpose 后内存不连续，contiguous 让它连续，view 才能用
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_k
        )

        # ============ 第5步：输出投影 ============
        # (batch, seq_q, d_model) → (batch, seq_q, d_model)
        # 把拼接的多头信息融合成最终输出
        output = self.W_o(attn_output)
        return output
```

### W_o 拼接后为什么不能省

拼接之后 512 维向量的结构是 `[头0的64维 | 头1的64维 | ... | 头7的64维]`，8 个头的输出只是物理上排在一起，彼此之间没有交互。W_o 做一次 512→512 的矩阵乘法，让输出每一维都能依赖所有头的信息——这就是"融合不同头"的含义。没有 W_o，输出第 0 维只来自头 0，第 65 维只来自头 1，无法跨头交互。

### 注意力输出向量的含义

这个向量是**该位置融合了上下文信息之后的新表示**。经过注意力前，"love" 的向量只包含自己的词义信息；经过注意力后，它融入了 "I" 和 "you" 的信息，变成了"在 I __ you 这个上下文中的 love"。不同注意力类型产出的向量含义不同：

```
编码器自注意力输出:  "在这个句子语境下的 love"
解码器掩蔽自注意力输出: "知道前面是'我'之后的'爱'"
解码器交叉注意力输出: "知道前面是'我'，并且对应源句'love'的'爱'"
```
单个词的嵌入向量是静态的——"bank" 在 "river bank" 和 "bank account" 里是同一个向量。经过注意力之后，"bank" 根据上下文变成了两个不同的向量，一个偏向"河岸"，一个偏向"银行"。每过一层编码器，向量就融入更多、更高层次的上下文信息。6 层叠加后，每个位置的 512 维向量已经编码了整个句子的语义。

---

## 5. 掩蔽多头注意力（Masked Multi-Head Attention）

![[causal_mask.png]]

### 作用

和多头注意力完全一样的计算，唯一区别是加了一个**因果掩码**：位置 `i` 只能看到位置 `0, 1, ..., i`，不能看到未来的位置 `i+1, i+2, ...`。

### 为什么需要它

解码器在推理（预测）时是**自回归**的：先生成第1个词，再用第1个词生成第2个词，以此类推。训练时虽然一次性输入了整个目标序列，但必须用 mask 模拟这个逐步生成的过程，否则模型会"偷看答案"。

### 掩码的形状

```
对于长度为 4 的序列，causal mask 是：
[[1, 0, 0, 0],    ← 位置0只能看自己
 [1, 1, 0, 0],    ← 位置1能看0和1
 [1, 1, 1, 0],    ← 位置2能看0、1、2
 [1, 1, 1, 1]]    ← 位置3能看所有
```

### 代码

```python
def generate_causal_mask(seq_len):
    """生成因果掩码（下三角矩阵）"""
    # torch.tril: 取下三角（包括对角线），上三角置 0
    # ones(seq_len, seq_len): 全 1 矩阵
    # 结果: (seq_len, seq_len)，1 表示可以看，0 表示不能看
    mask = torch.tril(torch.ones(seq_len, seq_len))
    # 增加 batch 和 head 维度: (1, 1, seq_len, seq_len)
    return mask.unsqueeze(0).unsqueeze(0)
```

使用时直接传给 `MultiHeadAttention` 的 `mask` 参数，和普通多头注意力用**同一个类**。

### 初始时 Q·K^T 有意义吗

没有。训练开始时 W_q 和 W_k 都是随机初始化的，注意力权重接近均匀分布。经过成千上万次梯度更新，W_q 和 W_k 逐渐学会让"应该关注"的词对点积变大。

---

## 6. 加 & 规范化（Add & Norm）

![[add_norm.png]]

### 作用

每个子层（注意力或前馈网络）的输出不直接传给下一层，而是先和输入做**残差连接**（相加），再做**层归一化**。

### 输入 / 输出

- **输入**: `x`（子层的输入）和 `sublayer(x)`（子层的输出），都是 `(batch, seq_len, d_model)`
- **输出**: `(batch, seq_len, d_model)` — 形状不变

### 层归一化公式

对每个样本的每个位置，取出它的 d_model 维特征向量 $x = [x_1, x_2, ..., x_d]$，做三步：

$$\mu = \frac{1}{d}\sum_{i=1}^{d}x_i$$

$$\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$$

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i + \beta_i$$

其中 $\epsilon$ 是一个极小值（如 $10^{-5}$），防止除以 0。$\gamma$ 和 $\beta$ 是可学习参数，形状都是 $(d,)$。

数值例子：

假设 d_model=4，某个位置的特征向量 $x = [2, 4, 6, 8]$：

**第1步：算均值**

$$\mu = \frac{2+4+6+8}{4} = 5$$

**第2步：算方差**

$$\sigma^2 = \frac{(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2}{4} = \frac{9+1+1+9}{4} = 5$$

**第3步：归一化**（假设 $\gamma=[1,1,1,1]$，$\beta=[0,0,0,0]$，即初始状态）

$$\hat{x}_1 = \frac{2-5}{\sqrt{5}} = \frac{-3}{2.236} = -1.34$$

$$\hat{x}_2 = \frac{4-5}{\sqrt{5}} = \frac{-1}{2.236} = -0.45$$

$$\hat{x}_3 = \frac{6-5}{\sqrt{5}} = \frac{1}{2.236} = 0.45$$

$$\hat{x}_4 = \frac{8-5}{\sqrt{5}} = \frac{3}{2.236} = 1.34$$

```
归一化前: [2,     4,     6,     8    ]  均值=5, 方差=5
归一化后: [-1.34, -0.45, 0.45,  1.34 ]  均值≈0, 方差≈1
```

γ 和 β 的作用：

归一化后均值=0、方差=1，但模型可能需要别的分布。$\gamma$ 和 $\beta$ 让模型自己学习最佳的缩放和偏移：

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i + \beta_i$$

```
如果模型学到 γ=[2,2,2,2], β=[1,1,1,1]:
  最终输出 = [-1.34×2+1, -0.45×2+1, 0.45×2+1, 1.34×2+1]
           = [-1.68,      0.10,      1.90,     3.68    ]
```

极端情况下如果 $\gamma_i = \sigma$，$\beta_i = \mu$，归一化就被完全撤销了——相当于模型说"这个位置不需要归一化"。所以 LayerNorm 给了模型一个选择权：默认归一化，但如果不合适，可以学着绕过去。

### 为什么需要残差连接

深层网络容易梯度消失。残差连接提供"直通通道"，梯度可以直接从输出传回输入。`output = LayerNorm(x + sublayer(x))`，如果 sublayer 什么都没学到（输出全0），网络至少不会变差。

### 为什么用层归一化而不是批归一化

假设一个 batch 有 3 个句子，每个句子 4 个位置，每个位置 3 维特征：

```
数据形状: (batch=3, seq=4, d_model=3)

句子0: [[a, b, c], [d, e, f], [g, h, i], [j, k, l]]
句子1: [[m, n, o], [p, q, r], [s, t, u], [v, w, x]]
句子2: [[A, B, C], [D, E, F], [G, H, I], [J, K, L]]
```

**批归一化**：固定一个位置和一个特征维度，沿 batch 方向算均值和方差：

```
比如"位置0的第0维特征"：
  句子0的值: a
  句子1的值: m
  句子2的值: A

用 (a, m, A) 这三个数算均值和方差，然后归一化
```

**层归一化**：固定一个样本和一个位置，沿特征维度算均值和方差：

```
比如"句子0的位置0"：
  特征0: a
  特征1: b
  特征2: c

用 (a, b, c) 这三个数算均值和方差，然后归一化
```

批归一化把**不同句子的同一个位置**拉到一起比较。问题在于：

```
句子0: "我    爱    你    <pad>"
句子1: "今天  天气  真    好"
句子2: "他    很    开心  <pad>"

位置1的词分别是: "爱"、"天气"、"很"
```

"爱"是动词、"天气"是名词、"很"是副词。把三者的特征拉到一起算均值方差，相当于强行要求**不同句子的同一位置的词应该有相似的统计特性**。但自然语言里位置 1 可以是任何词性、任何语义，这个假设根本不成立。

更糟的是句子长度不同，短句末尾是 `<pad>`：

```
位置3的词分别是: "<pad>"、"好"、"<pad>"

把一个有意义的"好"和两个无意义的 <pad> 一起归一化？
```

层归一化对**每个样本的每个位置独立做**：

```
句子0 位置1 "爱" 的特征: [0.3, -0.7, 0.5]
  → 只用这3个数算均值和方差，和其他句子无关

句子1 位置1 "天气" 的特征: [0.8, 0.2, -0.4]
  → 只用这3个数算均值和方差，和其他句子无关
```

每个词只和自己的特征维度比较，不需要假设"不同句子的同一位置有相似分布"。不管句子多长多短、位置上是什么词，都能正常归一化。

总结：

批归一化假设"不同样本的同一位置可以放在一起比较"，这在图像里成立（位置 (10,10) 在不同图片里大概率都是"某个物体的局部"），但在 NLP 里不成立（位置 3 在不同句子里可以是完全不同的词）。层归一化只看单个样本单个位置的特征维度，不做跨样本比较，所以不受这个问题影响。

### 代码

```python
class AddNorm(nn.Module):
    """残差连接 + 层归一化"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # nn.LayerNorm(d_model): 沿最后一维（特征维）做归一化
        # 对每个 (batch, seq_pos) 独立计算均值和方差
        # 然后 (x - mean) / std * gamma + beta
        # gamma 和 beta 是可学习参数，形状 (d_model,)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):
        # 1. 先对子层输出做 dropout（正则化）
        # 2. 加上原始输入（残差连接）
        # 3. 层归一化
        return self.layer_norm(x + self.dropout(sublayer_output))
```

**数值例子**：假设 `d_model=4`，某个位置的向量是 `[2, 4, 6, 8]`：

- 均值 = 5, 方差 = 5
- 归一化后 ≈ `[-1.34, -0.45, 0.45, 1.34]`
- 再乘 gamma 加 beta 得到最终值
---

## 7. 逐位前馈网络（Position-wise Feed-Forward Network）

![[ffn.png]]

### 作用

对每个位置独立做两层全连接变换。这是 Transformer 中唯一引入**非线性**的地方（靠 ReLU）。注意力层本身只做线性加权，非线性全靠这个 FFN。

### 输入 / 输出

- **输入**: `(batch, seq_len, d_model)` 例如 `(32, 10, 512)`
- **输出**: `(batch, seq_len, d_model)` — 形状不变

### 为什么叫"逐位"

**每个位置用相同的参数，但位置之间不交互**。本质是一个 `1×1 卷积`，或者说对 `seq_len` 个向量各自独立做同一个 MLP。

### 为什么需要它
注意力只做加权平均（线性操作），多层线性叠加等价于一层线性，深度白费。FFN 中的 ReLU（`max(0,x)`）引入非线性：选择性地"杀死"某些维度，不同输入被杀死的维度不同，实现了条件判断的效果。注意力负责"从哪里取信息"，FFN 负责"拿到信息后怎么处理"。

### 为什么需要升降维，而不是直接ReLU

如果直接对 512 维做 ReLU：

```
输入:  [0.3, -0.7, 0.5, -0.2, 0.8, ...]   512维
ReLU:  [0.3,  0,   0.5,  0,   0.8, ...]   512维
```

ReLU 只做一件事：负数变 0，正数不动。没有任何可学习参数，没有任何矩阵运算。它只是把大约一半的维度砍掉，**模型没有选择权**——哪些维度被砍掉完全取决于输入值的正负，而不是"这个信息有没有用"。

先升维再 ReLU 的效果完全不同：

```
输入:     [0.3, -0.7, 0.5]           3维 (简化示例)
          ↓
Linear1:  W₁ × 输入                  W₁ 是可学习的 (3→12) 矩阵
          ↓
中间:     [0.8, -1.2, 0.3, -0.5, 2.1, -0.9, 0.1, -1.5, 0.7, 0.4, -0.3, 1.1]  12维
          ↓
ReLU:     [0.8,  0,   0.3,  0,   2.1,  0,   0.1,  0,   0.7, 0.4,  0,   1.1]  12维
          ↓
Linear2:  W₂ × 中间                  W₂ 是可学习的 (12→3) 矩阵
          ↓
输出:     [0.6, 0.9, -0.2]           3维
```

关键区别在于 **Linear1 在 ReLU 之前重新组合了输入的各个维度**。原始的第 0 维的正负是固定的，但 Linear1 可以学到"把第 0 维和第 2 维相加减"生成一个新维度，这个新维度的正负代表了某种语义判断。ReLU 砍掉的不再是原始特征，而是 W₁ 学出来的**语义组合**。

升维到 4 倍的意义：

512 维直接 ReLU，只有 512 种"保留或砍掉"的组合。升维到 2048 再 ReLU，有 2048 个独立的"开关"，每个开关代表一种 W₁ 学出来的特征组合。

```
512维直接ReLU:   2^512 种开关模式，但每个开关绑死了一个原始维度
2048维升维ReLU:  2^2048 种开关模式，每个开关是 512 维的任意线性组合
```

这就像画画——512 支固定颜色的笔 vs 2048 支可以自由调色的笔。后者的表达力远远更强。

### 降维回 512 的意义

升维后有 2048 维，但后续模块（注意力层、残差连接）都期望 512 维输入。Linear2 把 ReLU 筛选后的 2048 维信息**压缩**回 512 维，相当于"从筛选结果中提炼出最有用的表示"。

整个过程可以理解为：**升维展开特征空间 → ReLU 做条件筛选 → 降维压缩精华**。三步缺一不可，缺了升维 ReLU 没有选择权，缺了降维维度对不上。
### 代码

```python
class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # 第一层: 升维 d_model → d_ff (512 → 2048)
        self.linear1 = nn.Linear(d_model, d_ff)
        # 第二层: 降维 d_ff → d_model (2048 → 512)
        self.linear2 = nn.Linear(d_ff, d_model)
        # ReLU 激活函数: max(0, x)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # 1. 升维 + ReLU: (batch, seq_len, d_model) → (batch, seq_len, d_ff)
        #    每个位置独立做，位置之间不交互
        x = self.relu(self.linear1(x))
        # 2. dropout 正则化
        x = self.dropout(x)
        # 3. 降维: (batch, seq_len, d_ff) → (batch, seq_len, d_model)
        x = self.linear2(x)
        return x
```

---

## 8. 编码器块（Encoder Block）

![[encoder_block.png]]


### 作用

编码器的一个完整层，包含两个子层：**多头自注意力 → Add&Norm → 前馈网络 → Add&Norm**。

### 输入 / 输出

- **输入**: `(batch, src_len, d_model)` + 可选的 padding mask
- **输出**: `(batch, src_len, d_model)` — 形状不变

### 为什么形状不变

残差连接要求子层输入和输出维度相同。形状不变意味着可以**任意堆叠**——论文用了 6 层。

## 代码
```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # 子层1: 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 子层2: 前馈网络
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        # 两个 Add&Norm
        self.addnorm1 = AddNorm(d_model, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)

    def forward(self, x, src_mask=None):
        # x: (batch, src_len, d_model)

        # 子层1: 自注意力
        # Q=K=V=x → 每个位置关注源序列中所有其他位置
        attn_output = self.self_attn(x, x, x, src_mask)
        # 残差连接 + 层归一化
        x = self.addnorm1(x, attn_output)

        # 子层2: 前馈网络
        ffn_output = self.ffn(x)
        # 残差连接 + 层归一化
        x = self.addnorm2(x, ffn_output)

        return x  # (batch, src_len, d_model)
```

---

## 9. 解码器块（Decoder Block）

![[decoder_block.png]]

### 作用

解码器的一个完整层，包含三个子层：

1. **掩蔽多头自注意力** — 目标序列自己关注自己（带因果掩码）
2. **编码器-解码器注意力（交叉注意力）** — 目标序列去关注源序列
3. **前馈网络**

每个子层后都有 Add&Norm。

### 输入 / 输出

- **输入**: `(batch, tgt_len, d_model)` + 编码器输出 `(batch, src_len, d_model)` + masks
- **输出**: `(batch, tgt_len, d_model)` — 形状不变

### 三个子层分别在干什么

| 子层     | Q 来自   | K, V 来自   | 作用               |
| ------ | ------ | --------- | ---------------- |
| 掩蔽自注意力 | 解码器前一层 | 解码器前一层    | 目标词之间互相看（但不能看未来） |
| 交叉注意力  | 解码器    | **编码器输出** | 目标词去"查询"源序列的信息   |
| FFN    | -      | -         | 逐位非线性变换          |

### 编码器和解码器各用几种注意力

编码器每一层只有 1 种注意力（自注意力：Q=K=V=x，每个源词看所有源词）。解码器每一层有 2 种注意力**依次串联**执行，不是选一个：

```
第1步 掩蔽自注意力:  Q=K=V=x（目标词之间互看，但加因果 mask 不能看未来）
  → "我前面已经生成了什么？"
第2步 交叉注意力:    Q=第1步的输出，K=V=enc_output（拿目标词去源句里查信息）
  → "源句里有什么和我相关的？"
第3步 FFN:          非线性变换
  → "拿到这些信息后怎么处理？"
```

顺序不能调换——如果先交叉再自注意力，"爱" 去源句查信息时还不知道前面是 "我"，查出来的结果缺少目标端上下文。三者底层都是同一个 `MultiHeadAttention` 类，只是传入的 Q/K/V 来源和 mask 不同：

```
                     Q 来自   K,V 来自   mask
编码器自注意力        x        x         无（或padding）
掩蔽自注意力          x        x         因果mask（下三角）
交叉注意力            x      enc_output   无（或padding）
              都进入同一个 scaled_dot_product_attention 函数
```

### 交叉注意力为什么关键
这是编码器和解码器之间唯一的信息通道。Q 来自解码器（当前生成状态的需求），K/V 来自编码器（源句信息）。没有它，解码器完全看不到源句，无法区分翻译 "I love you" 还是 "I hate you"。

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # 子层1: 掩蔽多头自注意力（代码和 MultiHeadAttention 一样，只是传入 causal mask）
        self.masked_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 子层2: 编码器-解码器交叉注意力
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 子层3: 前馈网络
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        # 三个 Add&Norm
        self.addnorm1 = AddNorm(d_model, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)
        self.addnorm3 = AddNorm(d_model, dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        """
        x:          (batch, tgt_len, d_model) — 解码器的输入
        enc_output: (batch, src_len, d_model) — 编码器的最终输出
        tgt_mask:   因果掩码，防止看到未来位置
        src_mask:   源序列的 padding 掩码
        """
        # ====== 子层1: 掩蔽自注意力 ======
        # Q=K=V=x，但用 tgt_mask 屏蔽未来位置
        attn1 = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.addnorm1(x, attn1)

        # ====== 子层2: 交叉注意力 ======
        # Q 来自解码器（当前层的输出），K 和 V 来自编码器的输出
        # 这让解码器的每个位置都能"查询"源序列的所有位置
        attn2 = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.addnorm2(x, attn2)

        # ====== 子层3: 前馈网络 ======
        ffn_output = self.ffn(x)
        x = self.addnorm3(x, ffn_output)

        return x  # (batch, tgt_len, d_model)
```

---

## 10. 全连接输出层（Final Linear + Softmax）

### 作用

把解码器最后一层的 d_model 维向量映射到 vocab_size 维，得到每个位置对词表中每个词的"分数"（logits），再通过 softmax 转成概率。

### 输入 / 输出

- **输入**: `(batch, tgt_len, d_model)`
- **输出**: `(batch, tgt_len, vocab_size)`

## 代码
```python
# 就是一个普通的 nn.Linear
output_layer = nn.Linear(d_model, vocab_size)
# 前向: logits = output_layer(decoder_output)  → (batch, tgt_len, vocab_size)
# 训练时用 CrossEntropyLoss（内部包含 softmax）
# 推理时用 softmax(logits, dim=-1) 取概率最大的词
```

---

## 11. 完整 Transformer 组装

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 n_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()

        # ====== 编码器部分 ======
        self.src_embed = TokenEmbedding(src_vocab_size, d_model)
        self.src_pos = PositionalEncoding(d_model, dropout)
        # 堆叠 num_layers 个编码器块
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # ====== 解码器部分 ======
        self.tgt_embed = TokenEmbedding(tgt_vocab_size, d_model)
        self.tgt_pos = PositionalEncoding(d_model, dropout)
        # 堆叠 num_layers 个解码器块
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # ====== 输出层 ======
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask=None):
        """编码器前向传播"""
        # 1. 词嵌入 + 缩放
        x = self.src_embed(src)        # (batch, src_len, d_model)
        # 2. 加上位置编码
        x = self.src_pos(x)            # (batch, src_len, d_model)
        # 3. 依次通过每个编码器层
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x                        # (batch, src_len, d_model)

    def decode(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        """解码器前向传播"""
        # 1. 词嵌入 + 缩放
        x = self.tgt_embed(tgt)        # (batch, tgt_len, d_model)
        # 2. 加上位置编码
        x = self.tgt_pos(x)            # (batch, tgt_len, d_model)
        # 3. 依次通过每个解码器层
        for layer in self.decoder_layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return x                        # (batch, tgt_len, d_model)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        完整的前向传播
        src: (batch, src_len)  — 源序列的词 ID
        tgt: (batch, tgt_len)  — 目标序列的词 ID
        """
        # 编码
        enc_output = self.encode(src, src_mask)
        # 解码
        dec_output = self.decode(tgt, enc_output, tgt_mask, src_mask)
        # 投影到词表
        logits = self.output_linear(dec_output)  # (batch, tgt_len, vocab_size)
        return logits
```

---

## 12. 设计动机总结

| 组件 | 解决什么问题 | 如果去掉会怎样 |
|------|-------------|--------------|
| 嵌入层 | 离散 ID → 连续向量 | 无法输入神经网络 |
| 位置编码 | 注入顺序信息 | "狗咬人"和"人咬狗"一样 |
| 多头自注意力 | 建模任意距离的依赖关系 | 退化为逐位 MLP，无法看其他位置 |
| 因果掩码 | 防止解码器偷看未来 | 训练时泄露答案，推理时不一致 |
| 交叉注意力 | 编码器→解码器信息通道 | 解码器看不到源序列 |
| FFN | 逐位非线性变换 | 纯线性，表达力不足 |
| 残差连接 | 缓解梯度消失 | 深层网络训不动 |
| 层归一化 | 稳定训练，加速收敛 | 训练不稳定，loss 震荡 |
| sqrt(d_k) 缩放 | 防止 softmax 饱和 | 注意力权重趋向 one-hot，梯度消失 |
| sqrt(d_model) 嵌入缩放 | 平衡嵌入和位置编码量级 | 位置信息淹没词义信息 |

---

## 13. 训练过程完整代码

![[training_loop.png]]

![[gradient_flow.png]]
### mask 生成

```python
def create_masks(src, tgt, src_pad_idx=0, tgt_pad_idx=0):
    """
    生成训练所需的两种 mask
    src: (batch, src_len) 源序列的词 ID
    tgt: (batch, tgt_len) 目标序列的词 ID
    """
    # ====== 源序列 padding mask ======
    # 源句中 pad 位置（ID=0）标记为 0，其他为 1
    # 例如 src = [5, 32, 7, 0, 0]，则 mask = [1, 1, 1, 0, 0]
    # unsqueeze 扩展维度以便广播到 (batch, 1, 1, src_len)
    # 这样在注意力计算中，所有 head、所有 query 位置都会屏蔽同样的 pad
    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)
    # 形状: (batch, 1, 1, src_len)

    # ====== 目标序列 mask = padding mask + 因果 mask ======
    # 第1部分: padding mask，和源序列一样
    tgt_pad_mask = (tgt != tgt_pad_idx).unsqueeze(1).unsqueeze(2)
    # 形状: (batch, 1, 1, tgt_len)

    # 第2部分: 因果 mask（下三角矩阵），防止看到未来
    tgt_len = tgt.size(1)
    causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device))
    # 形状: (tgt_len, tgt_len)
    # 例如 tgt_len=3:
    # [[1, 0, 0],
    #  [1, 1, 0],
    #  [1, 1, 1]]

    # 两个 mask 取交集（都为1才为1）
    # 既不能看 pad，也不能看未来
    tgt_mask = tgt_pad_mask & causal_mask.bool()
    # 形状: (batch, 1, tgt_len, tgt_len)

    return src_mask, tgt_mask
```

### 训练循环

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ====== 超参数 ======
src_vocab_size = 10000    # 源语言词表大小
tgt_vocab_size = 10000    # 目标语言词表大小
d_model = 512             # 隐藏维度
n_heads = 8               # 注意力头数
num_layers = 6            # 编码器/解码器各几层
d_ff = 2048               # FFN 中间维度
dropout = 0.1             # Dropout 概率
lr = 1e-4                 # 学习率
num_epochs = 20           # 训练轮数
src_pad_idx = 0           # 源语言 padding 的 ID
tgt_pad_idx = 0           # 目标语言 padding 的 ID

# ====== 创建模型 ======
model = Transformer(
    src_vocab_size, tgt_vocab_size, d_model,
    n_heads, num_layers, d_ff, dropout
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ====== 损失函数 ======
# CrossEntropyLoss 内部自动做 softmax，所以模型输出 logits 即可
# ignore_index=tgt_pad_idx: padding 位置不计入 loss
criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)

# ====== 优化器 ======
# Adam 比普通 SGD 收敛更快，适合 Transformer
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

# ====== 训练循环 ======
model.train()  # 切换到训练模式（启用 Dropout）

for epoch in range(num_epochs):
    epoch_loss = 0

    for batch_idx, (src, tgt) in enumerate(train_dataloader):
        # src: (batch, src_len)  源序列词 ID
        # tgt: (batch, tgt_len)  目标序列词 ID，例如 [<bos>, 我, 爱, 你, <eos>]
        src = src.to(device)
        tgt = tgt.to(device)

        # ====== 准备输入和标签 ======
        # 解码器输入: 去掉最后一个词（不需要输入 <eos>）
        # 例如 [<bos>, 我, 爱, 你]
        tgt_input = tgt[:, :-1]

        # 标签: 去掉第一个词（不需要预测 <bos>）
        # 例如 [我, 爱, 你, <eos>]
        tgt_label = tgt[:, 1:]
        # 这样 tgt_input 的每个位置对应预测 tgt_label 的那个位置：
        #   输入 <bos> → 预测 我
        #   输入 我    → 预测 爱
        #   输入 爱    → 预测 你
        #   输入 你    → 预测 <eos>

        # ====== 生成 mask ======
        src_mask, tgt_mask = create_masks(src, tgt_input, src_pad_idx, tgt_pad_idx)

        # ====== 前向传播 ======
        logits = model(src, tgt_input, src_mask, tgt_mask)
        # logits: (batch, tgt_len-1, vocab_size)
        # 每个位置输出词表大小的分数

        # ====== 计算 loss ======
        # CrossEntropyLoss 要求输入: (N, C) 和标签: (N,)
        # N = batch × (tgt_len-1)，C = vocab_size
        # 所以需要 reshape
        logits_flat = logits.reshape(-1, tgt_vocab_size)   # (N, vocab_size)
        labels_flat = tgt_label.reshape(-1)                 # (N,)
        loss = criterion(logits_flat, labels_flat)

        # ====== 反向传播 ======
        optimizer.zero_grad()  # 清空上一步的梯度（否则会累加）
        loss.backward()        # 计算 loss 对所有参数的梯度
                               # ∂loss/∂W_q, ∂loss/∂W_k, ∂loss/∂W_v, ∂loss/∂W_o,
                               # ∂loss/∂FFN参数, ∂loss/∂Embedding权重, ...
                               # 所有参数的梯度一次性全部算出

        # ====== 梯度裁剪（可选）======
        # 防止梯度爆炸：如果梯度的总范数超过 1.0，就等比缩小
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # ====== 更新参数 ======
        optimizer.step()       # 所有参数同时更新：
                               #   W_q  = W_q  - lr × ∂loss/∂W_q
                               #   W_k  = W_k  - lr × ∂loss/∂W_k
                               #   W_o  = W_o  - lr × ∂loss/∂W_o
                               #   ...每个参数都更新

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
```

### 所有参数怎么训练

没有"先训练 W_q 再训练 W_o"的说法。一次 `loss.backward()` 计算所有参数的梯度，一次 `optimizer.step()` 更新所有参数。整个前向传播是一条完整计算链，反向传播沿链从后往前走，链上每经过一个有参数的模块就计算该参数的梯度。这就是端到端训练。

---

## 14. 推理（测试/预测）过程完整代码

![[inference.png]]

训练时目标序列是已知的，一次性输入。推理时目标序列未知，必须**逐个生成**。

```python
def translate(model, src, src_pad_idx=0, bos_idx=1, eos_idx=2, max_len=50):
    """
    用训练好的模型翻译一个句子
    src: (1, src_len) 一个源句的词 ID
    bos_idx: <bos>（句子开始符）的 ID
    eos_idx: <eos>（句子结束符）的 ID
    max_len: 最长生成多少个词（防止无限循环）
    """
    model.eval()  # 切换到评估模式（关闭 Dropout）

    with torch.no_grad():  # 推理时不需要计算梯度，节省内存
        # ====== 编码器只跑一次 ======
        # 源句是固定的，编码一次就够了
        src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)
        enc_output = model.encode(src, src_mask)
        # enc_output: (1, src_len, 512)
        # 这个结果会被解码器的每一步反复使用（作为交叉注意力的 K 和 V）

        # ====== 解码器逐步生成 ======
        # 初始输入只有 <bos>
        generated = [bos_idx]

        for step in range(max_len):
            # 把已生成的词转成张量
            tgt_so_far = torch.tensor([generated], device=src.device)
            # 第1步: tgt_so_far = [[<bos>]]              形状 (1, 1)
            # 第2步: tgt_so_far = [[<bos>, 我]]           形状 (1, 2)
            # 第3步: tgt_so_far = [[<bos>, 我, 爱]]       形状 (1, 3)

            # 生成因果 mask
            tgt_len = tgt_so_far.size(1)
            tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=src.device))
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

            # 解码器前向传播
            dec_output = model.decode(tgt_so_far, enc_output, tgt_mask, src_mask)
            # dec_output: (1, tgt_len, 512)

            # 只取最后一个位置的输出（最新生成的位置）
            last_pos = model.output_linear(dec_output[:, -1, :])
            # last_pos: (1, vocab_size)

            # 取概率最大的词（贪心解码）
            next_word = last_pos.argmax(dim=-1).item()
            # 例如: next_word = 42，对应词表里的 "我"

            # 加入已生成序列
            generated.append(next_word)

            # 如果生成了 <eos>，停止
            if next_word == eos_idx:
                break

        return generated

# ====== 使用示例 ======
# 假设源句 "I love you" 的词 ID 是 [5, 32, 7]
src = torch.tensor([[5, 32, 7]], device=device)
result = translate(model, src)
# result 可能是 [1, 42, 88, 7, 2]
#                ↑   ↑   ↑  ↑  ↑
#              <bos> 我  爱  你 <eos>
# 然后用词表把 ID 转回文字即可
```

### 训练 vs 推理的关键区别

|  | 训练 | 推理 |
|--|------|------|
| 目标序列 | 已知，一次性全部输入 | 未知，逐个生成 |
| 因果 mask | 防止"偷看未来" | 天然看不到（还没生成）|
| 编码器 | 每个 batch 跑一次 | 跑一次，结果反复用 |
| Dropout | 开启 | 关闭 |
| 梯度 | 需要 | 不需要 (torch.no_grad) |

---
