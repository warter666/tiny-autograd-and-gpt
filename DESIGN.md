# 01-AI 设计文档：micrograd 重写 + picoGPT 重写

## 参照项目

### micrograd（karpathy/micrograd，17.5k★）
标量级反向自动微分引擎 + 极简神经网络库。核心：
- `Value`：标量值节点，记录 `data`、`grad`、父节点集合 `_prev`、产生它的运算 `_op`，以及一个闭包 `_backward`。
- 每个运算（add/mul/pow/relu）在前向时构造闭包，`backward()` 对输出节点做**递归拓扑排序**，然后逆序逐节点应用链式法则（梯度用 `+=` 累加）。
- `nn.py` 只有 Neuron/Layer/MLP 三层，参数是 `Value` 的列表。

### picoGPT（jaymody/picoGPT，3.5k★）
60 行 numpy 实现 GPT-2 推理。核心链路：
- `gpt2()`：token+position embedding → N × transformer block（pre-LN、多头因果注意力、GELU 前馈层）→ final LayerNorm → 权重绑定输出（`ln_f(x) @ wte.T`）。
- `generate()`：贪心解码，每步重算全序列前向。
- 参数直接从 OpenAI 的 TF checkpoint 读，命名保持一致（`c_attn`/`c_proj`/`ln_1`/`mlp` 等）。

## 重写目标与范围

### autograd/（micrograd 重写）
功能对齐：add/mul/pow/relu + 全部右操作符与派生运算符，`Value.backward()`。
**增强点**（与原版差异）：
1. **迭代式拓扑排序**（显式两阶段栈），原版递归实现在深图上会爆 Python 递归栈。
2. 补充 `exp` / `log` / `tanh` 算子——后续 minigpt 的数值验证和常见激活需要。
3. `_prev` 用 tuple 保持确定性遍历顺序。
4. 梯度验证双通道：与 PyTorch 数值对拍 + 中心差分校验。

### minigpt/（picoGPT 重写）
**范围**：picoGPT 只有推理+贪心解码；本重写补齐完整训练链路。
1. 前向：numpy 实现，逐 block 缓存中间量（`forward_with_cache`）供反向使用。
2. 反向：**手推全链路 numpy 反向传播**（linear / gelu(tanh近似) / layer_norm / 因果注意力 softmax / embedding scatter-add / 输出权重绑定的 wte 梯度合并），用中心差分逐参数校验。
3. 采样：temperature / top-k / top-p（原版只有 argmax）。
4. 训练：Adam + 下一 token 交叉熵，在合成循环序列上验证 loss 收敛与生成正确性。
5. 参数命名与 picoGPT/GPT-2 checkpoint 保持一致，保证将来可直接加载 124M 权重做推理。

## 验证策略

| 层级 | 方法 | 通过标准 |
|------|------|----------|
| autograd 算子 | 与 torch 同表达式对拍 | forward/grad 完全一致 |
| autograd 附加算子（exp/log/tanh） | 中心差分 | 相对误差 < 1e-6 |
| minigpt 每个参数 | 中心差分 vs 手推反向 | 相对误差 < 1e-4 |
| minigpt 采样 | 固定种子分布检查 | 合法概率、形状正确 |
| minigpt 端到端 | 合成循环序列训练 | loss 下降并生成正确循环 |

## 目录

```
01-ai/
├── DESIGN.md            本文档
├── autograd/            micrograd 重写
│   ├── engine.py        Value 引擎（迭代拓扑排序）
│   ├── nn.py            Module/Neuron/Layer/MLP
│   ├── test_engine.py   torch 对拍 + 差分校验
│   └── train_mlp_demo.py  合成数据训练演示
└── minigpt/             picoGPT 重写（推理→训练→采样）
    ├── model.py         前向 + 生成（temperature/top-k/top-p）
    ├── backward.py      手推反向传播
    ├── train.py         Adam 训练循环
    └── test_minigpt.py  差分梯度校验 + 采样/生成冒烟
```
