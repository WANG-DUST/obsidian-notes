# BAKU: An Efficient Transformer for Multi-Task Policy Learning

> **论文信息**：Siddhant Haldar, Zhuoran Peng, Lerrel Pinto (NYU)
> **发表时间**：arXiv 2024.06，Preprint
> **项目主页**：[baku-robot.github.io](https://baku-robot.github.io)

---

## 一句话总结

BAKU 通过系统性 ablation，将 FiLM-conditioned 视觉编码器 + decoder-only Transformer trunk + Action Chunking + 可插拔 Action Head 精心组合成一个简洁的多任务策略学习架构，在 129 个仿真任务和 30 个 xArm 真机任务上大幅超越 MT-ACT 和 RT-1（LIBERO-90 上绝对提升 36%，从 54% → 90%）。

---

## 要解决的问题

- 多任务策略往往显著落后于单任务专属策略。
- 机器人数据采集昂贵，靠"堆数据"不是可持续路径。
- 现有多任务架构（MT-ACT、RT-1、PerceiverActor 等）设计复杂，各组件的必要性不清晰——没有人在统一框架下做过干净的消融对比。
- BAKU 不追求发明新算法，而是做一次系统性架构消融，把有效的设计点组合成一个 clean 的架构。核心贡献是"选择与验证"，novelty 在于组合。

---

## 核心架构概览

BAKU 的整体 pipeline 分为三个模块：

```
输入: {多视角图像 O_t, 本体状态 s_t, 任务指令 T (文本或goal image)}
       │
       ▼
[Sensory Encoders]  ── ResNet-18+FiLM (共享视觉编码器)
                    ── 2-layer MLP (proprioception)
                    ── MiniLM (文本, 冻结)
                    ── 各模态投影到同一维度 d=256
       │
       ▼
[Observation Trunk]  ── Causal Transformer Decoder (minGPT, 8层4头, d=256)
                     ── 每个模态作为一个 token
                     ── 追加可学习的 action token
       │
       ▼
[Action Head]  ── MLP / GMM / BeT / VQ-BeT / Diffusion (可插拔)
       │
       ▼
输出: 动作块 {a_t, a_{t+1}, ..., a_{t+k-1}} (拼接为一个向量一次性预测)
```

训练时每次只输入当前 1 帧观测（无历史），预测未来 k 步的动作 chunk（LIBERO 上 k=10）。

---

## 模块一：Sensory Encoders

### 视觉分支（ResNet-18 + FiLM）

对每一路相机图像 `(B, 3, 128, 128)`：

```
image → ResNet-18 (with FiLM) → (B, 512) → Linear(512, 256) → (B, 256)
```

所有相机视角**共享同一个 ResNet-18**（Table 8 ablation 显示独立编码器仅 +2pp，但每加一路相机参数量涨 15%，不划算）。

两路图像 concat 到 batch 维度过一次 forward：

```
imgs_stacked: (2B, 3, 128, 128) → ResNet-FiLM → (2B, 512) → reshape → (B, 2, 512) → project → (B, 2, 256)
```

得到 2 个视觉 token，每个 256 维。

### FiLM 详解

**FiLM 全称 Feature-wise Linear Modulation（逐通道线性调制）**，核心目的是让**同一张图片在不同任务下产出不同的视觉特征**。

#### 数学定义

给定文本 embedding $z \in \mathbb{R}^{d_z}$（MiniLM 输出，$d_z = 384$），对第 $l$ 个 FiLM 层：

$$\gamma_l = W_l^\gamma\, z + b_l^\gamma \in \mathbb{R}^{C_l}$$
$$\beta_l = W_l^\beta\, z + b_l^\beta \in \mathbb{R}^{C_l}$$
$$\text{FiLM}(x)_{b,c,h,w} = \gamma_{l,c} \cdot x_{b,c,h,w} + \beta_{l,c}$$

各符号含义：

| 符号 | 类型 | 含义 |
|------|------|------|
| $l$ | 下标 | FiLM 层的索引（ResNet-18 共 8 个 BasicBlock → 8 个 FiLM 层） |
| $z$ | $\mathbb{R}^{384}$ | 条件向量（MiniLM 文本 embedding） |
| $C_l$ | 标量 | 第 $l$ 层的通道数（stage 1/2/3/4 分别是 64/128/256/512） |
| $W_l^\gamma$ | $\mathbb{R}^{C_l \times 384}$ | 生成 $\gamma$ 的可学习权重 |
| $b_l^\gamma$ | $\mathbb{R}^{C_l}$ | 生成 $\gamma$ 的可学习偏置 |
| $\gamma_{l,c}$ | 标量 | 第 $c$ 个通道的缩放系数 |
| $\beta_{l,c}$ | 标量 | 第 $c$ 个通道的平移量 |
| $x$ | $\mathbb{R}^{B \times C_l \times H \times W}$ | 该层被调制的特征图 |

γ 和 β 沿 spatial 维度 broadcast——一个通道内所有像素位置共享同一个 γ 和 β。

#### 为什么是"逐通道"而不是"逐像素"

CNN 的同一个通道在整张特征图上代表同一类语义特征（比如"红色区域""竖直边缘""杯子把手形状"等）。逐通道调制等于逐语义调制：$\gamma_{l,c}$ 控制"通道 $c$ 代表的语义特征"整体被放大还是抑制。如果做逐像素调制，参数量会爆炸且破坏 CNN 的空间平移不变性。

#### FiLM 在 ResNet-18 中的插入位置

```
BasicBlock 内部:
        ┌── Conv3x3 + BN + ReLU + Conv3x3 + BN ── FiLM(cond=z) ──┐
input ──┤                                                          + ── ReLU ── output
        └────────────── (identity / shortcut) ─────────────────────┘
```

FiLM 插在**每个 BasicBlock 的最终 ReLU 之前、BN 之后**。ResNet-18 共 8 个 BasicBlock，所以有 8 个独立的 FiLM 层。全部 FiLM 层引入的参数约 1.48M。

#### FiLM 对梯度的影响

反向传播时，从 FiLM 输出传回 ResNet 特征图的梯度：

$$\frac{\partial \mathcal{L}}{\partial x_{b,c,h,w}} = \frac{\partial \mathcal{L}}{\partial \text{FiLM}(x)_{b,c,h,w}} \cdot \gamma_{l,c}$$

梯度被 $\gamma_{l,c}$ 乘了。这意味着：
- $\gamma_{l,c}$ 大 → 该通道梯度被放大 → ResNet 在该通道上更新更快、学得更精细
- $\gamma_{l,c} \approx 0$ → 梯度被压制 → 该通道对当前任务不重要，不用花精力优化
- $\gamma_{l,c} < 0$ → 梯度反号 → 特征方向被翻转

因果方向是：FiLM 通过 γ 告诉网络"这个通道对当前任务很重要"→ 梯度自然就大 → 该通道被训练得更好。

**注意**：MiniLM 是冻结的，所以梯度在文本 embedding $z$ 上截断——只更新 FiLM 层的 $W^\gamma, b^\gamma, W^\beta, b^\beta$ 权重和 ResNet 本身的权重。

#### FiLM vs 无 FiLM 的对比

**无 FiLM（如 MT-ACT）**：ResNet 是任务无关的。不管指令是"抓杯子"还是"抓碗"，同一张图产出同一个特征向量。任务信息要到 Transformer trunk 里才跟视觉融合。

**有 FiLM（BAKU）**：ResNet 的每一层都通过 FiLM 接收文本条件。同一张图在不同指令下，ResNet 输出的特征向量不同——视觉编码本身就具备任务感知能力，Transformer 的融合负担更轻。

BAKU 是 FiLM + late fusion 双路并用：FiLM 让视觉编码器任务感知，text token 进 Transformer 让 trunk 层面也有任务信息。消融显示 FiLM 带来 +3pp（LIBERO-90 上 87→90），边际收益不算大但稳定。

#### FiLM 的初始化注意事项

- **γ 偏置初始化为 1**：因为 FiLM 是乘法调制，如果 γ 初始化接近 0，特征全被压缩为 0，梯度接近 0，网络学不动。偏置初始化为 1 让训练初期 γ≈1, β≈0，等价于恒等映射。
- **FiLM 放在 BN 之后**：BN 已经归一化过，再做仿射变换更稳定。放在 BN 之前会被 BN 部分抵消。
- **每层独立参数**：每个 BasicBlock 一套独立的 $W^\gamma, b^\gamma, W^\beta, b^\beta$，不共享。

#### 最小实现代码

```python
class FiLMLayer(nn.Module):
    def __init__(self, num_channels, cond_dim):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, num_channels)
        self.to_beta  = nn.Linear(cond_dim, num_channels)
    
    def forward(self, x, z):
        # x: (B, C, H, W), z: (B, cond_dim)
        gamma = self.to_gamma(z).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta  = self.to_beta(z).unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        return gamma * x + beta
```

### 本体分支

```
state: (B, 9) → Linear(9, 256) → ReLU → Linear(256, 256) → (B, 256)
```

得到 1 个 proprio token，256 维。

### 文本分支

```
text → MiniLM (frozen) → (B, 384) → Linear(384, 256) → (B, 256)
```

得到 1 个 text token，256 维。MiniLM 权重冻结不参与训练。同一个文本 embedding 有两个用处：（1）产生 FiLM 里的 γ/β；（2）自己作为一个 token 送进 trunk。

### Sensory Encoders 汇总

经过三个分支，手上有 4 个 256 维 tokens：

```
[text_token, view1_token, view2_token, proprio_token]  shape: (B, 4, 256)
```

---

## 模块二：Observation Trunk（Causal Transformer Decoder）

### 为什么选 Decoder-only 而不是 Encoder-Decoder

ACT 用标准 encoder-decoder：encoder 处理 obs，decoder 用 k 个 position embedding 作 query 通过 cross-attention 解码出 k 个动作。

BAKU 做了关键简化——扔掉 encoder，只留 decoder。因为 BAKU 把"预测 k 步动作"完全交给了 action head（一个 MLP 从单个向量回归出整个 chunk），所以 trunk 不需要 k 个 query 分别解码，只需要输出一个好的 action feature。

### 三种 Transformer 架构的区分

| 架构 | Mask 类型 | 核心结构 |
|------|-----------|----------|
| Encoder-only（如 BERT） | Full（双向） | Self-Attention → FFN |
| Encoder-Decoder（如原始 Transformer, ACT） | Encoder: Full; Decoder: Causal + **Cross-Attention** | 编码+解码两阶段 |
| Decoder-only（如 GPT, BAKU） | Causal（单向） | Self-Attention → FFN，**没有 Cross-Attention** |

BAKU 叫"decoder-only"是因为它用了 causal mask，这是原始 Transformer decoder 的标志性特征。但它**没有 cross-attention**，因为没有独立的 encoder 需要交叉查询。它把 obs tokens 和 action token 放在同一个序列里，用 self-attention 统一处理——self-attention 同时完成了"理解"和"为生成做准备"两件事。

### 输入 token 序列的构造

在 4 个 obs token 后面 append 一个**可学习的 action token**（`nn.Parameter`，shape (256,)）：

| Token 位置 | 内容          | 来源                        |
| -------- | ----------- | ------------------------- |
| 0        | text_tok    | MiniLM → Linear           |
| 1        | view1_tok   | ResNet-FiLM(相机1) → Linear |
| 2        | view2_tok   | ResNet-FiLM(相机2) → Linear |
| 3        | proprio_tok | MLP(state)                |
| 4        | action_tok  | nn.Parameter (可学习)        |

拼成序列 $X^{(0)} \in \mathbb{R}^{B \times 5 \times 256}$。

action token 类似 BERT 的 [CLS]——它一开始不携带具体信息，但经过 attention 后从其他 token 里"读取"信息，最终成为面向动作预测的汇聚表征。它不是隐变量，不依赖 GT action，不是随机的，训练和推理时行为完全一致。

### 位置编码

minGPT 用**可学习的绝对位置编码**（不是 sinusoidal）：

$$\text{pos\_emb} \in \mathbb{R}^{1 \times 5 \times 256}$$

直接加到输入 token 上：$X^{(0)} \leftarrow X^{(0)} + \text{pos\_emb}$。

**"绝对"和"可学习"描述的是两个不同维度**：
- "绝对"= 每个位置有自己专属的向量，不随输入内容变化
- "可学习"= 这个向量的数值由梯度下降优化出来，而不是手工设定

位置编码跟 token 的对应关系纯粹靠 `torch.cat` 的拼接顺序决定——第 $i$ 个位置编码加到序列中第 $i$ 个 token 上，没有任何智能匹配。

**位置编码不仅是身份标签，还编码了"位置之间的先验交互关系"**。因为位置编码是通过加法融入 token 的，它会直接参与 attention score 的计算。展开 attention score：

$$\text{score}(i,j) = \underbrace{(W_Q x_i)^\top(W_K x_j)}_{\text{内容-内容}} + \underbrace{(W_Q x_i)^\top(W_K p_j)}_{\text{内容-位置}} + \underbrace{(W_Q p_i)^\top(W_K x_j)}_{\text{位置-内容}} + \underbrace{(W_Q p_i)^\top(W_K p_j)}_{\text{位置-位置}}$$

其中最后一项"位置-位置"完全不依赖输入内容，提供位置之间的固定 attention 偏好。例如网络可以学到让 view1 和 view2 的编码很接近（因为都是视觉），让 action_token 的编码跟其他都不同（因为角色独特）。

### Causal Mask

5 个 token 位置的 mask 矩阵（$M_{ij} = 1$ 表示位置 $i$ 能 attend 位置 $j$）：

$$M = \begin{pmatrix} 1 & 0 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 & 0 \\ 1 & 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 1 & 1 \end{pmatrix}$$

action token（位置 4）能看到所有 token，没有 obs token 能看到 action token。

在无历史的默认配置下，这个 causal mask 的约束实际上有些多余——5 个 token 全来自同一时间步，没有因果性需要保护。full attention 理论上不会更差。BAKU 用 causal mask 主要是因为直接复用了 minGPT 的实现。

如果用历史观测（BAKU 消融过但最终不用），token 序列变长，causal mask 才真正发挥作用——保证当前时刻的 action token 不能看到未来时刻的信息。

### 单个 Transformer 层的前向

minGPT 用 **pre-norm**（先 LayerNorm 再 Attention/FFN），不是 post-norm（先 Attention/FFN 再 LayerNorm）。

```
输入 X^(l) → LayerNorm → Multi-Head Self-Attention (causal) → + (residual) → X_mid
X_mid      → LayerNorm → FFN (256→1024→256, GELU)           → + (residual) → X^(l+1)
```

#### Pre-norm vs Post-norm

Pre-norm 训练更稳定，原因在于梯度流的差异：

**Post-norm**：$X^{(l+1)} = \text{LN}(X^{(l)} + \text{Attn}(X^{(l)}))$。梯度从最后一层传回第 1 层要穿过 L 个 LayerNorm，每个都会缩放梯度，连乘后不可控。

**Pre-norm**：$X^{(l+1)} = X^{(l)} + \text{Attn}(\text{LN}(X^{(l)}))$。残差连接在 LayerNorm 之外，梯度求导时出现单位矩阵 $I$：

$$\frac{\partial X^{(l+1)}}{\partial X^{(l)}} = I + \frac{\partial \text{Attn}(\text{LN}(X^{(l)}))}{\partial X^{(l)}}$$

$I$ 提供一条"高速公路"，梯度可以不衰减地直达第 1 层。即使 Attn/FFN 分支梯度出问题，直通路径保证浅层仍能收到合理信号。

#### Multi-Head Self-Attention 细节

4 个 head，$d_{\text{head}} = 256 / 4 = 64$：

$$Q = \tilde{X} W_Q, \quad K = \tilde{X} W_K, \quad V = \tilde{X} W_V \quad \in \mathbb{R}^{B \times 5 \times 256}$$

拆成 $(B, 4, 5, 64)$ 后计算 attention：

$$A = \frac{Q K^\top}{\sqrt{d_{\text{head}}}}, \quad \text{apply causal mask}, \quad \alpha = \text{Softmax}(A)$$

$$\text{head\_out}_i = \sum_j \alpha_{i,j} \cdot V_j$$

拼接多头 + 输出投影：$\text{attn\_out} = \text{Concat}(\text{heads}) \cdot W_O$

残差连接：$X_{\text{mid}} = X^{(l)} + \text{attn\_out}$

#### FFN

$$\text{FFN}(x) = W_2 \, \text{GELU}(W_1 \, x + b_1) + b_2$$

- $W_1 \in \mathbb{R}^{1024 \times 256}$：扩展到 4 倍
- $W_2 \in \mathbb{R}^{256 \times 1024}$：压回
- GELU 激活（不是 ReLU）
- 对每个 token 独立、参数共享地作用（position-wise）

> 如果要在 BAKU 上加 MoE，就是替换这个 FFN 为 MoE FFN——每个 token 根据路由器选择不同的 expert FFN。

### Trunk 输出

经过 8 层后取 action token 位置的输出：

$$\text{action\_feat} = X^{(8)}[:, 4, :] \in \mathbb{R}^{B \times 256}$$

### Trunk 参数量

每层约 0.79M（attention 投影 + FFN + LayerNorm），8 层共约 6.3M，加位置编码和最终 LayerNorm 共约 **6.5M**。

### 梯度回传

虽然 loss 只直接连在 action token（位置 4），但通过 attention 反向传播，所有 obs token 都能收到梯度：

$$\frac{\partial \mathcal{L}}{\partial V_j} = \alpha_{4,j} \cdot \frac{\partial \mathcal{L}}{\partial \text{attn\_out}_4}$$

被关注得多的 token 收到的梯度更大。梯度从 action token 通过 attention "广播"到所有 obs token，再回传到各自的 sensory encoder。

---

## 模块三：Action Head

Action head 接收 trunk 输出的 256 维向量，产出整个动作 chunk。BAKU 的核心设计之一是将 action head 从 trunk 解耦——换 head 不需要改 trunk，只要新 head 接受 256 维输入、输出 (k, d_a) 的 chunk 即可。

### MLP Head（默认）

```python
head = nn.Sequential(
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, k * d_a),   # 256 → 70
)
# 输出 reshape 成 (B, 10, 7)
```

整个 chunk 被拼成一个 70 维向量一次性回归。reshape 就是把长度 70 的一维向量按顺序切成 10 段，每段 7 个数（前 7 个 = 第 1 步的 Δxyz + Δrpy + gripper，第 8-14 个 = 第 2 步，以此类推）。网络通过 loss 的监督信号自然学会了每个输出神经元对应哪一步的哪个维度。

Loss 为标准 L2：

$$\mathcal{L} = \frac{1}{B \cdot k} \sum_{b=1}^{B} \sum_{i=0}^{k-1} \| a^{gt}_{b,i} - \hat{a}_{b,i} \|^2$$

优点：极简、推理快、参数少。缺点：只能输出单峰预测——同一 obs 对应多种正确动作时会回归到平均值。仿真 demo 通常单峰，够用。

### GMM Head（高斯混合模型，M=5）

预测 M 个高斯分量的均值 $\mu_m$、标准差 $\sigma_m$ 和混合权重 $\pi_m$：

$$p(\hat{a}|h) = \sum_{m=1}^{M} \pi_m \cdot \mathcal{N}(\hat{a};\, \mu_m,\, \text{diag}(\sigma_m^2))$$

Loss 为负对数似然（NLL）。推理时按 π 采样分量再从对应高斯采样。LIBERO-90 上 0.84，比 MLP 的 0.90 低 6pp——LIBERO demo 本身单峰，GMM 的额外参数增加了优化难度。

### BeT Head（Behavior Transformer，64 clusters）

预训练阶段：对所有 GT chunk 做 k-means 聚类得到 64 个中心。训练时分类选 bin + 回归 offset 修正。Loss = Focal loss（分类）+ L2（offset）。推理时选 bin 加 offset。

LIBERO-90 上 0.89，接近 MLP。

### VQ-BeT Head（Vector-Quantized BeT）

用 Residual VQ-VAE（两层 codebook，大小 16 和 256）替代 k-means。第一层选"大致方向"，第二层修正"细节偏差"。

训练分两阶段：（1）离线训练 VQ-VAE；（2）端到端训练策略，分类预测两层 code + 连续 offset。

Loss = CE（两层分类）+ L2（offset）。

**在真机上比 MLP 好 5pp**（0.91 vs 0.86），因为人类 demo 天然多峰——VQ 的不同码字捕获不同行为模式，输出的永远是一条"真实存在的"路径而非平均路径。

### Diffusion Head（2 层 Transformer diffusion）

把动作预测建模为去噪过程。训练时对 GT chunk 加噪，网络学预测噪声。推理时从纯噪声逐步去噪。

Loss = 噪声预测 L2。推理最慢（每步需要 N 次 forward）。Meta-World 上崩到 0.45。

### 五种 Head 对比表

| Head | 多峰能力 | 预处理 | Loss | 推理速度 | LIBERO-90 | 真机 |
|------|---------|--------|------|---------|-----------|------|
| MLP | 无 | 无 | L2 | 最快 | **0.90** | 0.86 |
| GMM | 有 | 无 | NLL | 快 | 0.84 | — |
| BeT | 有 | k-means | Focal+L2 | 快 | 0.89 | — |
| VQ-BeT | 有 | VQ-VAE | CE+L2 | 快 | **0.90** | **0.91** |
| Diffusion | 有 | 无 | L2 | 最慢 | 0.89 | — |

---

## 训练流程

### 输入输出

- **输入**：当前时刻 $t$ 的多模态观测（图像 + 本体状态 + 文本指令）
- **输出**：未来 k 步的**动作增量**（不是未来的状态）

动作增量 $(\Delta x, \Delta y, \Delta z, \Delta roll, \Delta pitch, \Delta yaw, \text{gripper})$，机器人收到后执行 $\text{state}_{t+1} = \text{state}_t + \text{action}_t$。

### 训练数据来源

从专家演示轨迹中取样。一条轨迹是人类遥操时同步记录的 obs 和 action 序列。随机选时刻 $t$，取 obs\_t 作为输入，取 $[a_t, a_{t+1}, \ldots, a_{t+k-1}]$ 作为标签。

### Loss 函数

多任务 BC 目标（论文 Eq.2）：

$$\mathcal{L}^{BC} = \mathbb{E}_{(o^e, a^e, g^e) \sim \mathcal{T}^e} \| a^e - \pi^{BC}(o^e | g^e) \|^2$$

- $\mathcal{T}^e = \{(o_t, a_t)_{t=0}^T\}_{n=0}^N$：专家轨迹集合
- $o^e$：专家观测
- $a^e$：专家动作
- $g^e$：goal 变量（文本或 goal image）

实际训练就是标准监督学习——拿专家演示的真实动作序列当标准答案，让网络的预测尽量逼近它。

### 前向与反向梯度路径

```
loss → MLP head → action token (位置4) → Transformer trunk (8层)
                                              │
              ├── attention 梯度广播到所有 obs tokens
              │      ├── text_tok → proj_text → MiniLM (冻结, 截断)
              │      ├── vis_toks → proj_vis → ResNet-FiLM → 更新 CNN 权重
              │      │                              ↑
              │      │                         FiLM γ/β MLP 也被更新
              │      └── proprio_tok → mlp_state → 更新 MLP 权重
              └── action_token → 更新 nn.Parameter 本身
```

### 训练超参

| 参数 | 值 |
|------|------|
| Learning rate | 1e-4 |
| Optimizer | Adam |
| Batch size | 64 (LIBERO/Meta-World/xArm), 128 (DMC) |
| Training steps | 600k (LIBERO/Meta-World), 200k (xArm), 2M (DMC) |
| Chunk length | 10 (LIBERO/Meta-World), 20 (xArm), 3 (DMC) |
| 总参数量 | ~10M (encoders 2.1M + trunk 6.5M + head 1.4M) |

---

## 推理流程：Temporal Ensembling

### 原理

每一步都 query 一次策略（不是每 k 步一次），得到重叠的 chunk 预测。对当前时刻 $t$，手上有来自过去若干步预测的多个候选动作，用指数加权平均集成：

$$w_i = \exp(-m \cdot i)$$

- $i$：该预测的"老旧程度"（$i=0$ 最老）
- $m$：越小越信任新预测

归一化后加权平均：$a_t^{\text{execute}} = \sum_i w_i a_t^{(\tau_i)} / \sum_i w_i$

temporal ensembling 只增加推理开销，训练时完全没有额外成本。作用是大幅平滑动作——任何一步的抖动预测都被其他预测拉回来。

---

## 实验结果

### 主要结果（Table 1）

| Method | LIBERO-90 (90 tasks) | Meta-World (30 tasks) | DMC (9 tasks) | Real Robot |
|--------|-----|-----|-----|-----|
| RT-1 | 0.16 | 0.65 | 0.66 | 0.37 |
| MT-ACT | 0.54 | 0.13 | 0.59 | 0.56 |
| BAKU (MLP) | **0.90** | **0.79** | **0.70** | 0.86 |
| BAKU (VQ-BeT) | 0.90 | 0.78 | 0.70 | **0.91** |

### Ablation 关键结论（Table 3）

1. **Trunk 类型**：Transformer > MLP，LIBERO-90 上 +9pp（81→90）
2. **模型尺寸**：4.4M/10M/31M 表现接近；114M 在 LIBERO-90 上崩到 0.19（过拟合）。10M 是甜蜜点
3. **Action Head**：仿真 MLP 够用；真机 VQ-BeT +5pp（多峰 demo 需要多模态建模）
4. **Action Chunking**：manipulation +14pp，locomotion −4pp（只对操作任务好用）
5. **Observation History**：last-step loss 灾难（−36pp）；multi-step loss 拉回但无提升→单帧就够
6. **Goal Modality**：文本/goal image/中间 goal image 差不多
7. **FiLM**：+3pp，边际收益
8. **共享 vs 独立视觉编码器**：独立仅 +2pp 但参数涨 15%/相机
9. **Trunk 输入方式**：多 token 分开 > concat 成一个向量（LIBERO-90 上 +3pp）

### 关于 Observation History 的细节

BAKU 最终只用 1 帧（无历史）。消融实验中用历史时，会为每个时间步添加独立的 action token：

- **last-step loss**：只对最后一个 action token 算 loss → 中间 token 梯度稀疏，性能崩到 0.54
- **multi-step loss**：对每个 action token 都算 loss（给 Transformer 更丰富的训练梯度）→ 性能恢复到 0.90，但跟不用历史一样

结论：历史帧本身对 BAKU 的任务没有信息增益。历史 action token 的作用不是类似 ACT 的隐变量 $z$——它只是信息汇聚的载体，不编码动作风格、不依赖 GT action、不是随机变量。

---

## BAKU vs ACT 对照表

| 维度 | ACT / MT-ACT | BAKU |
|------|-------------|------|
| Trunk 架构 | Encoder-Decoder | Decoder-only |
| Chunk 预测方式 | k 个 position query，cross-attention 逐个解码 | 1 个 action token → 1 个向量 → MLP 一次回归 k·d_a |
| CVAE | 有（学习 style latent z） | 无 |
| 多模态动作建模 | CVAE 在 trunk 级别处理 | 交给可插拔的 action head 处理 |
| 文本条件方式 | Task embedding 作为 token | Token + FiLM 双路 |
| 视觉编码器 | ResNet（无条件） | ResNet + FiLM（条件化） |
| Action head | 固定回归 | 可插拔 5 种 |
| 模型大小 | ~80M | ~10M |

---

## Novelty

1. **组合与验证**：首次在统一受控框架下对多任务策略学习的各设计组件做干净的消融实验
2. **解耦架构**：将 action head 从 observation trunk 明确剥离，使任何 SOTA action generation model 可即插即用

更接近一篇 rigorous engineering/benchmarking paper 而非 method paper，但 ablation 表对同方向研究有极高参考价值。

---

## 局限性

### 作者承认的

1. 精密操作任务失败（如开烤箱门）——多难度任务的数据共享可能伤害精密技能
2. 不能 chain skills——只能做单一 skill
3. 未研究 task 数量增加带来的 generalization emergence

### 额外观察

1. 114M 模型崩溃原因未深究（过拟合 or 超参？）——MoE 的 sparse activation 可能是解决方案
2. 10 rollouts/task 统计噪声大，无多 seed 结果
3. VQ-BeT 真机+5pp 但仿真无收益——LIBERO 上可能测不出多模态 head 的真实价值
4. MT-ACT 在 Meta-World 上只有 0.13 显得过低，比较不完全公平
5. 未对比 RVT、Octo、π0 等同期/后续 VLA 工作

---

## 完整训练伪代码

```python
class BAKU(nn.Module):
    def __init__(self, k=10, action_dim=7, d=256):
        super().__init__()
        self.text_encoder = MiniLM()                       # frozen
        self.resnet = ResNetFiLM_18(cond_dim=384)          # trainable
        self.proj_vis  = nn.Linear(512, d)
        self.proj_text = nn.Linear(384, d)
        self.mlp_state = nn.Sequential(nn.Linear(9, d), nn.ReLU(), nn.Linear(d, d))
        self.action_token = nn.Parameter(torch.randn(1, 1, d))
        self.trunk = CausalTransformer(d=d, n_layer=8, n_head=4)
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, k * action_dim)
        )

    def forward(self, imgs, state, text):
        B = imgs.shape[0]
        with torch.no_grad():
            text_emb = self.text_encoder(text)             # (B, 384)
        text_tok = self.proj_text(text_emb).unsqueeze(1)   # (B, 1, d)
        
        imgs_flat = imgs.reshape(B*2, 3, 128, 128)
        text_emb_2 = text_emb.repeat_interleave(2, dim=0)
        vis_feat = self.resnet(imgs_flat, cond=text_emb_2) # (2B, 512)
        vis_toks = self.proj_vis(vis_feat).reshape(B, 2, -1)
        
        prop_tok = self.mlp_state(state).unsqueeze(1)
        act_tok = self.action_token.expand(B, 1, -1)
        
        tokens = torch.cat([text_tok, vis_toks, prop_tok, act_tok], dim=1)
        out = self.trunk(tokens)
        action_feat = out[:, -1]
        return self.head(action_feat).reshape(B, 10, 7)

# Training loop
model = BAKU().cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
for step in range(600_000):
    batch = next(loader)
    pred = model(batch.imgs, batch.state, batch.text)
    loss = F.mse_loss(pred, batch.action_gt)
    opt.zero_grad(); loss.backward(); opt.step()
```
