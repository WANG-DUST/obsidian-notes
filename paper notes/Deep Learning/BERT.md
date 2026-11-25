
## 1. BERT 是什么 + 为什么这样设计

**一句话定位**：BERT = Transformer 的 **Encoder 部分**堆叠 N 层 + **双向**自注意力 + 两个**自监督预训练任务**。

和标准 Transformer / ACT 对比：

| 维度 | 标准 Transformer | ACT | BERT |
|---|---|---|---|
| 结构 | Encoder + Decoder | Encoder + Decoder (CVAE) | 只有 Encoder |
| 注意力方向 | Encoder 双向, Decoder 因果 | Encoder 双向, Decoder 因果 | **完全双向** |
| 任务 | 翻译等 seq2seq | 动作序列生成 | MLM + NSP 预训练 |

### 1.1 为什么只用 Encoder——从"自回归生成"和"因果 mask"说起

先搞懂两个概念。

**自回归生成**：想象你在一个字一个字地写作文：

> 今天 → 天气 → 真 → 好 → ...

你写"天气"的时候，只能看到前面已经写了"今天"，不能偷看后面还没写的内容。写"真"的时候，只能看到"今天天气"——永远是从左往右，一个接一个。这就是"自回归"：每一步的输出，依赖前面所有步的输出，像流水线一样一个个往外蹦。GPT 就是这样工作的。

**因果 mask**："因果"在这里不是哲学意义，而是一个技术限制——后面的字不能影响前面的字，就像时间的因果律，未来不能影响过去。具体实现上就是一个三角形的遮罩：

```
          今天  天气  真  好
今天       ✓    ✗    ✗   ✗    ← "今天"只能看自己
天气       ✓    ✓    ✗   ✗    ← "天气"能看"今天"和自己
真         ✓    ✓    ✓   ✗    ← "真"能看前三个
好         ✓    ✓    ✓   ✓    ← "好"能看所有前面的
```

✗ 的位置在注意力分数上加 -10000，softmax 之后就约等于 0——等于看不见。

**BERT 为什么不要这套东西？** 关键在于任务目标不同：

- **GPT 像考试时写作文**：你只能从第一个字写到最后一个字，不能偷看后面的内容。所以必须用因果 mask，否则作弊了。
- **BERT 像考试时做阅读理解**：整篇文章都摆在你面前，你来回看才能理解每个词在上下文里什么意思。

BERT 的目标是学**文本表示**（每个 token 的上下文向量），不需要自回归生成，所以不要 Decoder 的因果 mask，让每个位置能看到左右两侧所有 token——这就是 "Bidirectional" 的含义。注意力矩阵变成全 ✓，没有三角形限制：

```
          今天  ████  真  好
今天       ✓    ✓    ✓   ✓
████       ✓    ✓    ✓   ✓    ← 被 mask 的词也能看全部上下文
真         ✓    ✓    ✓   ✓
好         ✓    ✓    ✓   ✓
```

一句话总结：**要生成 → 必须因果 mask（不能偷看后文）→ 自回归 → GPT；要理解 → 不需要因果 mask（故意看全文）→ 双向 → BERT**。

在 ACT 里用的 Encoder 也是全 ✓ 模式（图像 token 之间互相都能看到），Decoder 才有那个三角形 mask（预测动作序列时不能偷看未来时刻的动作）。同一个道理。

### 1.2 为什么需要 MLM

如果直接用语言模型（预测下一个词）训练双向 Transformer，会出现"信息泄露"——每个位置可以通过自注意力看到自己想预测的词。MLM 通过**随机遮盖 15% 的 token，让模型预测被遮的词**，逼模型从上下文学到有用表示。

BERT 的训练方式是：给你一句完整的话，随机挖掉几个词（盖住），让你猜被盖住的是什么：

```
今天 [MASK] 真好 → 猜 [MASK] 是"天气"
```

要猜对"天气"，你既要看前面的"今天"，也要看后面的"真好"。如果只让你看左边"今天"两个字，太多可能性了。但加上右边的"真好"，就很容易推断是"天气"。所以 BERT 故意不加因果 mask，让每个位置都能看到左右两边所有词。

### 1.3 为什么还要 NSP

许多下游任务（问答、自然语言推理）涉及**两个句子的关系**，所以加一个判断"句子B是不是句子A的下一句"的二分类任务。后续研究如 RoBERTa 证明 NSP 帮助不大，但原始 BERT 有它。

---

## 2. 核心维度缩写速查

BERT 里反复出现 6 个字母缩写，先一次性搞清：

**B = Batch Size（批大小）**：一次喂给模型几个句子。比如 B=2 就是同时处理 2 个句子。GPU 擅长并行，一次算一批更快。

**L = Length（序列长度）**：一个句子有几个 token。比如 `[CLS] 今天 天气 真 好 [SEP]` 分词后 6 个 token，L=6。BERT 最长支持 L=512。不够长用 `[PAD]` 补齐，超长截断。

**H = Hidden Size（隐藏层维度）**：每个 token 被表示成一个多少维的向量。BERT-base 用 H=768，意思是每个 token 是一个 768 维的向量。可以类比：一张图是 3 通道（RGB），一个 token 是 768 "通道"。维度越大，模型越能装信息，但也越吃显存。

**A = Attention Heads（注意力头数）**：多头注意力里有几个头。BERT-base 用 A=12。每个头独立学一种关注模式——比如第 1 个头可能学语法关系，第 3 个头可能学指代关系，最后把 12 个头的结果拼起来。

**D = Head Dimension（每个头的维度）**：D = H / A = 768 / 12 = 64。把 768 维的向量平均切成 12 份，每份 64 维，每个头分到 64 维去做自己的注意力。最后 12 个头 × 64 维 = 768 维拼回来。可以这样想：H 是一整条蛋糕，A 是切几刀，D 是每块多大。

**V = Vocab Size（词表大小）**：词表里一共有多少个词。BERT 用 WordPiece 分词，英文版词表有 V=30522 个子词。MLM 任务最后要预测被遮住的词是哪个——就是从这 30522 个候选里选一个，所以 MLM 头的输出维度是 V。

BERT-base 默认值：`B=自定义, L=512, H=768, A=12, D=64, V=30522`

---

## 3. 整体数据流向

```
input_ids, token_type_ids  
       │
       ▼
┌─────────────────┐
│  Embeddings     │  Token + Position + Segment 三个 embedding 相加
└─────────────────┘
       │  (B, L, H)
       ▼
┌─────────────────┐  ┐
│  Encoder Layer 1│  │
├─────────────────┤  │ N 层堆叠
│  ...            │  │ (BERT-base: N=12, BERT-large: N=24)
├─────────────────┤  │
│  Encoder Layer N│  │
└─────────────────┘  ┘
       │  (B, L, H)
       ├───────────────────┐
       ▼                   ▼
   sequence_output      pooled_output  ← 取 [CLS] 位置过一个 Linear+Tanh
   (B, L, H)            (B, H)
       │                   │
       ▼                   ▼
   MLM Head             NSP Head
   预测每个被mask词     预测IsNext/NotNext
   (B, L, V)            (B, 2)
```

每个 Encoder Layer 内部：
```
x ──► Multi-Head Self-Attention ──► Dropout ──► Add(x) ──► LayerNorm ──► x'
x' ──► Linear(H→4H) ──► GELU ──► Linear(4H→H) ──► Dropout ──► Add(x') ──► LayerNorm ──► output
```

用 B=2, L=8 举例，数据流过模型时形状怎么变：
```
input_ids:        (2, 8)           ← 2个句子，每个8个token，都是整数id
                    │
                    ▼ Embedding查表
hidden_states:    (2, 8, 768)      ← 每个id变成768维向量
                    │
                    ▼ 拆成12个头
Q, K, V:          (2, 12, 8, 64)   ← 12个头，每个头64维
                    │
                    ▼ Q @ K^T
attention scores: (2, 12, 8, 8)    ← 8个token两两之间的相关性
                    │
                    ▼ softmax → @ V → 合并12个头
after attention:  (2, 8, 768)      ← 又回到768维
                    │
                    ▼ 12层Encoder之后
sequence_output:  (2, 8, 768)      ← 每个token的最终表示
                    │
                    ▼ MLM头
mlm_logits:       (2, 8, 30522)    ← 每个位置对30522个词的打分
```

---

## 4. 形状追踪表

设 `B=2, L=8, H=768, A=12, D=64, V=30522`：

| 阶段 | 张量 | 形状 |
|---|---|---|
| 输入 | `input_ids` | (2, 8) |
| 嵌入后 | `hidden_states` | (2, 8, 768) |
| Q/K/V 投影后 | `Q, K, V` | (2, 8, 768) → 拆头 → (2, 12, 8, 64) |
| 注意力分数 | `scores` | (2, 12, 8, 8) |
| 注意力输出 | `attn_out` | (2, 12, 8, 64) → 合头 → (2, 8, 768) |
| FFN 中间层 | `intermediate` | (2, 8, 3072) |
| Encoder 输出 | `sequence_output` | (2, 8, 768) |
| Pooler 输出 | `pooled_output` | (2, 768) |
| MLM logits | `mlm_logits` | (2, 8, 30522) |
| NSP logits | `nsp_logits` | (2, 2) |

---

## 5. PyTorch 实现

### 5.1 配置类

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BertConfig:
    """BERT 的所有超参数集中在这里，方便修改和传递"""
    def __init__(
        self,
        vocab_size=30522,           # 词表大小 V，WordPiece 分词后约 3 万
        hidden_size=768,            # 隐藏层维度 H，base 版 768，large 版 1024
        num_hidden_layers=12,       # Encoder 层数 N
        num_attention_heads=12,     # 多头注意力的头数 A，要能整除 hidden_size
        intermediate_size=3072,     # FFN 中间层维度，通常 = 4 * hidden_size
        hidden_dropout_prob=0.1,    # 全连接层后的 dropout 概率
        attention_probs_dropout_prob=0.1,  # 注意力权重的 dropout 概率
        max_position_embeddings=512,  # 最长序列长度 L 的上限，超过会报错
        type_vocab_size=2,          # segment embedding 的种类数，BERT 只有 2 类（句子A/B）
        layer_norm_eps=1e-12,       # LayerNorm 的 epsilon，防止除零
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
```

### 5.2 Embeddings 层

**作用**：把离散的 token id 变成连续的稠密向量，并注入位置信息和句子归属信息。

**为什么三个 embedding 相加**：Transformer 的自注意力本身不感知位置（置换不变），所以必须把位置编码加进去；BERT 处理句子对时还需要告诉模型"这个 token 属于句子A还是B"，所以加 segment embedding。三者相加是最简单的融合方式。

**Position Embedding 为什么是可学习的而不是 sin/cos**：论文里说两种试了效果差不多，可学习的写代码更简单就用了。代价是不能外推到训练时没见过的更长序列（sin/cos 理论上可以）。

```python
class BertEmbeddings(nn.Module):
    """Token + Position + Segment 三种 embedding 相加，然后 LayerNorm + Dropout"""
    def __init__(self, config):
        super().__init__()
        # 词嵌入：把 token id 映射为 H 维向量。padding_idx=0 让 pad token 的梯度为零
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        # 位置嵌入：可学习的位置编码（原 Transformer 用 sin/cos，BERT 改成可学习）
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # Segment 嵌入：0 表示句子 A，1 表示句子 B
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        # LayerNorm：对最后一维做归一化，稳定训练
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout：防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 预先注册一个 0..max_position-1 的张量作为 buffer，前向时直接 slice 用
        # register_buffer 让它跟着模型 .to(device)，但不算作参数（不更新）
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1))  # shape (1, max_len)
        )

    def forward(self, input_ids, token_type_ids=None):
        # input_ids: (B, L)
        seq_length = input_ids.size(1)
        
        # 取出本次需要的位置 id，shape (1, L)，靠 broadcast 自动扩展到 B
        position_ids = self.position_ids[:, :seq_length]
        
        # 如果用户没传 token_type_ids，默认全是 0（单句任务）
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # 三个 embedding 都是 (B, L, H)
        word_emb = self.word_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)
        tok_type_emb = self.token_type_embeddings(token_type_ids)
        
        # 直接相加（因为都是 H 维向量，语义是"位置/段落信息是对 token 表示的偏移"）
        embeddings = word_emb + pos_emb + tok_type_emb
        
        # LayerNorm 稳定数值，Dropout 防过拟合
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # (B, L, H)
```

**数据流向**：`(B, L)` → 三个 lookup，每个 `(B, L, H)` → 相加 → LN → Dropout → `(B, L, H)`

### 5.3 Multi-Head Self-Attention

**作用**：让每个 token 根据相关性"读取"序列里所有其他 token 的信息，形成上下文表示。

**为什么多头**：单头注意力只能捕捉一种关系模式；多头让模型在不同子空间并行学习多种关系（语法、共指、语义角色等），再拼接。

```python
class BertSelfAttention(nn.Module):
    """多头自注意力，实现 Scaled Dot-Product Attention"""
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0, \
            "hidden_size 必须能被 num_attention_heads 整除"
        
        self.num_heads = config.num_attention_heads      # A = 12
        self.head_dim = config.hidden_size // config.num_attention_heads  # D = 64
        self.all_head_size = self.num_heads * self.head_dim  # = H = 768
        
        # Q、K、V 三个线性投影，各自把 (B, L, H) 投到 (B, L, H)
        # 一开始就投到完整的 H 维，后面再拆成多头，这样比 A 个独立 Linear 高效
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """把 (B, L, H) 重塑成 (B, A, L, D)，即拆头 + 把头维度提前"""
        # x.size()[:-1] 是 (B, L)，再拼上 (A, D)，得到 (B, L, A, D)
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)          # (B, L, A, D)
        # permute 让形状变成 (B, A, L, D)，方便后面对 L×L 做矩阵乘
        return x.permute(0, 2, 1, 3)    # (B, A, L, D)

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: (B, L, H)
        
        # Q、K、V 投影 + 拆头
        Q = self.transpose_for_scores(self.query(hidden_states))  # (B, A, L, D)
        K = self.transpose_for_scores(self.key(hidden_states))    # (B, A, L, D)
        V = self.transpose_for_scores(self.value(hidden_states))  # (B, A, L, D)
        
        # 注意力分数: Q @ K^T，在最后两维做矩阵乘
        # (B, A, L, D) @ (B, A, D, L) -> (B, A, L, L)
        # 含义：scores[b, a, i, j] = 第 b 个样本第 a 个头中 token i 对 token j 的相关性
        scores = torch.matmul(Q, K.transpose(-1, -2))
        
        # 缩放：除以 sqrt(D)，防止维度大时 softmax 梯度消失
        scores = scores / math.sqrt(self.head_dim)
        
        # 加 attention_mask：padding 位置加 -10000（softmax 后约等于 0）
        # mask 形状: (B, 1, 1, L)，靠 broadcast 加到 (B, A, L, L) 上
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax 把分数变成概率分布
        attn_probs = F.softmax(scores, dim=-1)  # (B, A, L, L)
        attn_probs = self.dropout(attn_probs)
        
        # 加权求和：概率 @ V
        # (B, A, L, L) @ (B, A, L, D) -> (B, A, L, D)
        context = torch.matmul(attn_probs, V)
        
        # 合头: (B, A, L, D) -> (B, L, A, D) -> (B, L, H)
        context = context.permute(0, 2, 1, 3).contiguous()  # contiguous 让内存连续，view 才能用
        new_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_shape)  # (B, L, H)
        return context
```

**关键技巧**：`attention_mask` 不是直接乘 0/1，而是对要屏蔽的位置**加 -10000**。这样 softmax 之后这些位置接近 0，而保留的位置正常归一化。这个 mask 是从 `(B, L)` 的 padding mask 扩展成 `(B, 1, 1, L)` 的（后面 BertModel 里会处理）。注意 BERT 不用因果 mask（三角形 mask），所有真实 token 互相都能看到。

### 5.4 Self-Output（注意力后的残差 + LN）

```python
class BertSelfOutput(nn.Module):
    """注意力输出后的线性投影 + Dropout + 残差 + LayerNorm"""
    def __init__(self, config):
        super().__init__()
        # 这个线性层是 Transformer 论文里多头注意力最后那个 W^O
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # hidden_states: 注意力的输出 (B, L, H)
        # input_tensor:  注意力的输入（残差连接用）(B, L, H)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # 残差 + LN（BERT 用 Post-LN，先加后归一）
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
```

打包注意力子层：

```python
class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)  # 注意残差用的是原始输入
        return attention_output
```

### 5.5 Feed-Forward Network

**作用**：对每个位置独立做一次"升维 → 非线性 → 降维"的处理，给模型加入非线性表达能力。

**为什么 4×**：论文里中间层是 hidden 的 4 倍（3072），这是经验值；直觉上自注意力是"信息混合"，FFN 是"信息加工"，加工需要更大的容量。

**为什么 GELU 而不是 ReLU**：GELU 在 0 附近平滑，经验上 NLP 任务比 ReLU 略好。

```python
class BertIntermediate(nn.Module):
    """FFN 第一层：线性升维 + GELU"""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        # (B, L, H) -> (B, L, 4H)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """FFN 第二层：线性降维 + Dropout + 残差 + LN"""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # (B, L, 4H) -> (B, L, H)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # 残差用的是 FFN 的输入（即注意力的输出）
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
```

### 5.6 完整 Encoder Layer + 堆叠

12 层 Encoder 每一层学到的东西不一样。一般规律：浅层偏表面（词形、局部句法），中层偏句法/句子结构，深层偏语义/任务特定信息。这也是为什么微调时不冻结任何层效果通常最好。

```python
class BertLayer(nn.Module):
    """一个完整的 Transformer Encoder 层"""
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        # 子层 1: 自注意力 + 残差 + LN
        attention_output = self.attention(hidden_states, attention_mask)
        # 子层 2: FFN + 残差 + LN
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    """N 层 BertLayer 堆叠"""
    def __init__(self, config):
        super().__init__()
        # ModuleList 而不是 list，这样 PyTorch 才能把子模块的参数注册进来
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states  # (B, L, H)
```

### 5.7 Pooler

**作用**：从 `[CLS]` token 的输出再抽取一个"句子级"表示，给 NSP 和下游分类任务用。

`[CLS]` 能代表整句是因为自注意力让它能看到序列里所有 token，经过 12 层后它的表示就是所有 token 信息的某种聚合。但本身没什么神奇——是 NSP 任务训练它"专门用于句子级判断"才让它对句子级任务有用。Pooler 只是再加一层 Linear+Tanh 让它更适合分类。

```python
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # hidden_states: (B, L, H)
        # 取每个样本第 0 个位置（即 [CLS]）的表示
        first_token_tensor = hidden_states[:, 0]  # (B, H)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output  # (B, H)
```

### 5.8 BertModel：拼装主干

```python
class BertModel(nn.Module):
    """BERT 的主干：Embeddings + Encoder + Pooler"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # input_ids: (B, L)
        # attention_mask: (B, L)，1 表示真实 token，0 表示 padding
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # 把 (B, L) 的 mask 扩展成自注意力能用的 (B, 1, 1, L)
        # 1 -> 0（不屏蔽）；0 -> -10000（屏蔽）
        extended_attention_mask = attention_mask[:, None, None, :]  # (B, 1, 1, L)
        extended_attention_mask = (1.0 - extended_attention_mask.float()) * -10000.0
        
        # 走 embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)  # (B, L, H)
        # 走 Encoder
        sequence_output = self.encoder(embedding_output, extended_attention_mask)  # (B, L, H)
        # 走 Pooler
        pooled_output = self.pooler(sequence_output)  # (B, H)
        
        return sequence_output, pooled_output
```

### 5.9 预训练头（MLM + NSP）

```python
class BertPredictionHeadTransform(nn.Module):
    """MLM 头里的一个小变换：Linear + GELU + LN（原论文有这一步）"""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    """MLM 头：把每个位置的表示映射到词表上的 logits"""
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        # 输出 vocab_size 维：每个位置对每个词的得分
        # bias=False，因为我们手动加 bias，这样可以让 weight 和 word embedding 共享（参数 tying）
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 单独的 bias 参数
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # 把 decoder 的 bias 指向同一个 Parameter
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)  # (B, L, V)
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    """打包 MLM 头 + NSP 头"""
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        # NSP 是二分类
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)   # (B, L, V)  MLM
        seq_relationship_score = self.seq_relationship(pooled_output)  # (B, 2)  NSP
        return prediction_scores, seq_relationship_score
```

**参数共享技巧（weight tying）**：MLM 头的 decoder 权重通常和输入 word_embeddings 共享（`decoder.weight = word_embeddings.weight`），这样既省参数又让"输入→隐表示"和"隐表示→词"互为逆变换。在 BertForPreTraining 的 `__init__` 末尾加一句：

```python
self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight
```

### 5.10 BertForPreTraining + 损失

```python
class BertForPreTraining(nn.Module):
    """预训练模型：BertModel + MLM 头 + NSP 头 + 损失计算"""
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        # 权重共享
        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids,                # (B, L)
        attention_mask=None,      # (B, L)
        token_type_ids=None,      # (B, L)
        masked_lm_labels=None,    # (B, L)，非 mask 位置标 -100，MLM 标签
        next_sentence_label=None, # (B,)，NSP 标签（0 = IsNext, 1 = NotNext）
    ):
        sequence_output, pooled_output = self.bert(
            input_ids, attention_mask, token_type_ids
        )
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        
        total_loss = None
        if masked_lm_labels is not None and next_sentence_label is not None:
            # ignore_index=-100：非 mask 位置的标签设为 -100，loss 自动跳过它们
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # MLM loss：把 (B, L, V) reshape 成 (B*L, V)，标签 reshape 成 (B*L,)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                masked_lm_labels.view(-1)
            )
            # NSP loss
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2),
                next_sentence_label.view(-1)
            )
            # 两个损失直接相加（等权重）
            total_loss = masked_lm_loss + next_sentence_loss
        
        return {
            "loss": total_loss,
            "prediction_logits": prediction_scores,
            "seq_relationship_logits": seq_relationship_score,
        }
```

---

## 6. MLM 数据准备

很多人卡在这里。MLM 不是简单地把 15% 的位置全换成 `[MASK]`，而是有 80/10/10 策略：

- 选中 15% 的 token 作为预测目标
- 其中 **80%** 替换成 `[MASK]`
- **10%** 替换成随机词
- **10%** 保持原样

**为什么这么设计**：微调阶段不会有 `[MASK]` token，如果训练时只见到 `[MASK]`，模型会"偷懒"——只在 `[MASK]` 位置努力，其他位置不学。80/10/10 让模型不知道哪个位置真的被改了，被迫对每个位置都学好表示。

```python
def create_mlm_labels(input_ids, vocab_size, mask_token_id, pad_token_id, special_token_ids):
    """
    input_ids: (B, L) 原始 token id
    返回 (masked_input_ids, labels):
      masked_input_ids：替换后的输入
      labels：被预测位置是原始 id，其他位置是 -100
    """
    labels = input_ids.clone()
    # probability matrix: 每个位置以 0.15 概率被选中
    probability_matrix = torch.full(labels.shape, 0.15)
    
    # 特殊 token（[CLS], [SEP], [PAD]）不参与 mask
    special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_token_ids:
        special_tokens_mask |= (input_ids == sid)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    # 伯努利采样，得到 (B, L) 的 0/1 mask，1 表示被选中
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # 没被选中的位置，label 设为 -100，loss 忽略
    labels[~masked_indices] = -100
    
    # 80% 替换成 [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id
    
    # 10% 替换成随机词（在被选中且没被换成 [MASK] 的剩余里再 50%）
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]
    
    # 剩下 10% 保持原样（不需要做任何事）
    
    return input_ids, labels
```

NSP 数据准备就简单：50% 概率取真实下一句（label=0），50% 概率取随机另一篇文档的句子（label=1），拼接成 `[CLS] sentA [SEP] sentB [SEP]`，segment_ids 是 `[0,0,...,0,1,1,...,1]`。

---

## 7. 训练流程

```python
def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()  # 开启 dropout、BN 训练模式
    total_loss = 0.0
    
    for step, batch in enumerate(dataloader):
        # batch 是 dict，里面是上面预处理好的张量
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        mlm_labels = batch["mlm_labels"].to(device)
        nsp_labels = batch["nsp_labels"].to(device)
        
        # 1) 清梯度
        optimizer.zero_grad()
        
        # 2) 前向
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            masked_lm_labels=mlm_labels,
            next_sentence_label=nsp_labels,
        )
        loss = outputs["loss"]
        
        # 3) 反向
        loss.backward()
        
        # 4) 梯度裁剪（可选，防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 5) 更新参数
        optimizer.step()
        scheduler.step()  # learning rate 调度
        
        total_loss += loss.item()
        if step % 100 == 0:
            print(f"step {step}, loss {loss.item():.4f}")
    
    return total_loss / len(dataloader)
```

**优化器和调度器**：原论文用 AdamW + warmup + linear decay，典型配置：
- AdamW: lr=1e-4, β1=0.9, β2=0.999, weight_decay=0.01
- Warmup: 前 10000 步线性从 0 升到峰值 lr，之后线性衰减到 0
- BERT-base 在 16 块 TPU 上训练 4 天，语料是 BookCorpus + Wikipedia（约 33 亿词）

---

## 8. 前向 / 反向传播流程

### 8.1 前向（以 B=2, L=8 为例）

1. `input_ids (2,8)` → `Embeddings` → `(2,8,768)`
2. `(2,8,768)` → 第 1 层 Encoder：
   - Q/K/V 投影并拆头 → 3 × `(2,12,8,64)`
   - Q@K^T / √D → `(2,12,8,8)` → + mask → softmax → dropout
   - @V → `(2,12,8,64)` → 合头 → `(2,8,768)`
   - 线性投影 → dropout → +残差 → LN → `(2,8,768)`
   - FFN: Linear → GELU → Linear → dropout → +残差 → LN → `(2,8,768)`
3. 重复 12 层
4. `sequence_output (2,8,768)` → MLM 头 → `(2,8,30522)`
5. `pooled_output (2,768)` → NSP 头 → `(2,2)`
6. CrossEntropy 计算两个 loss，相加得 total_loss

### 8.2 反向

PyTorch 自动微分，`loss.backward()` 沿着计算图反向传播。梯度流的关键路径：

```
loss
 ├── (NSP path) → seq_relationship 线性层 → pooler → [CLS] 位置的 sequence_output
 └── (MLM path) → MLM 头 → 每个被 mask 位置的 sequence_output
        │
        ▼
   Encoder Layer 12 → ... → Encoder Layer 1
        │
        ▼
   Embeddings（word + position + segment）
```

由于残差连接，深层的梯度可以"短路"直接传到浅层，这是 Transformer 能堆 12+ 层不退化的核心原因。LayerNorm 又稳定了每层的激活分布。这两个机制和 ACT 里的完全一样。

---

## 9. 测试 / 推理流程

### 9.1 预训练阶段的验证

在验证集上跑 forward，记录 MLM accuracy 和 NSP accuracy：

```python
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()  # 关闭 dropout
    total_mlm_correct, total_mlm_count = 0, 0
    total_nsp_correct, total_nsp_count = 0, 0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        mlm_labels = batch["mlm_labels"].to(device)
        nsp_labels = batch["nsp_labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            masked_lm_labels=mlm_labels,
            next_sentence_label=nsp_labels,
        )
        mlm_logits = outputs["prediction_logits"]   # (B, L, V)
        nsp_logits = outputs["seq_relationship_logits"]  # (B, 2)
        
        # MLM: 只在 mlm_labels != -100 的位置算准确率
        mlm_preds = mlm_logits.argmax(dim=-1)  # (B, L)
        mask = (mlm_labels != -100)
        total_mlm_correct += ((mlm_preds == mlm_labels) & mask).sum().item()
        total_mlm_count += mask.sum().item()
        
        # NSP
        nsp_preds = nsp_logits.argmax(dim=-1)
        total_nsp_correct += (nsp_preds == nsp_labels).sum().item()
        total_nsp_count += nsp_labels.size(0)
    
    return total_mlm_correct / total_mlm_count, total_nsp_correct / total_nsp_count
```

### 9.2 下游任务微调

把 `BertModel` 取出来当 backbone，加一个任务相关的头：

- **文本分类**：用 `pooled_output (B, H)` → Linear → softmax
- **命名实体识别**：用 `sequence_output (B, L, H)` → Linear → 每个 token 一个标签
- **问答（SQuAD）**：用 `sequence_output` → 两个 Linear，分别预测答案起始位置和结束位置

微调时所有参数都解冻，用小学习率（2e-5 到 5e-5）训 2-4 个 epoch 通常就够了。

### 9.3 BERT 和 ACT 里 Transformer Encoder 的区别

两者结构几乎一样，差异在：
1. BERT 用可学习 Position Embedding，ACT 通常用 sin/cos 或 RoPE
2. BERT 是 Post-LN（残差后归一），很多新模型改成了 Pre-LN（归一后注意力）
3. 输入语义不同——ACT 输入的是图像特征+本体感觉，BERT 是离散 token id 