# 01-AI — 从零重写 micrograd 与 picoGPT

[![CI](https://github.com/warter666/tiny-autograd-and-gpt/actions/workflows/ci.yml/badge.svg)](https://github.com/warter666/tiny-autograd-and-gpt/actions/workflows/ci.yml)

两个包，全部 numpy + 标准库。torch 仅作为测试对拍基准（未安装时相关断言自动跳过）。

## autograd/ — 标量级自动微分引擎（重写 karpathy/micrograd，17.5k★）

`Value` 记录 `data / grad / _prev / _op / _backward`，前向时构造闭包，
`backward()` 做拓扑排序后逆序应用链式法则。相对原版的三处改动：

1. **迭代式拓扑排序**（显式两阶段栈）——原版递归实现在深图上会爆 Python 递归栈。
2. 补 `exp / log / tanh` 算子，供后续 minigpt 的数值验证与激活使用。
3. `_prev` 用 tuple，遍历顺序确定，梯度累加可复现。

`nn.py` 提供 Module / Neuron / Layer / MLP 三层抽象。

## minigpt/ — GPT-2 全链路（重写 jaymody/picoGPT，3.5k★）

picoGPT 只有推理和贪心解码；这里补齐到「前向 → 反向 → 采样 → 训练」的完整闭环。

- `model.py`：pre-LN transformer block、多头因果注意力、GELU(tanh 近似)、权重绑定的输出层。
  参数命名与 GPT-2 checkpoint 完全一致（`wte/wpe/c_attn/c_proj/ln_1/ln_2/mlp/ln_f`），
  将来可**直接加载 124M 权重**做推理。采样支持 temperature / top-k / top-p。
- `backward.py`：**手推全链路反向传播**——linear、GELU、LayerNorm、因果注意力 softmax、
  embedding scatter-add、权重绑定处 wte 的梯度合并。
- `train.py`：Adam + 下一 token 交叉熵。

## 验证

```bash
python -m autograd.test_engine
python -m autograd.train_mlp_demo
python -m minigpt.test_minigpt
python -m minigpt.train
```

| 层级 | 方法 | 通过标准 |
|------|------|----------|
| autograd 算子 | 与 torch 同表达式对拍 | forward/grad 完全一致 |
| autograd 附加算子 | 中心差分 | 相对误差 < 1e-6 |
| minigpt 每个参数 | 中心差分 vs 手推反向 | 相对误差 < 1e-4 |
| minigpt 采样 | 固定种子分布检查 | 概率合法、形状正确 |
| minigpt 端到端 | 合成循环序列（"abc…"）训练 | loss 下降且能生成正确循环 |

实测输出：
```
all engine tests passed
torch forward parity passed
gradient check passed on 140 entries, loss=2.3829
all minigpt tests passed
```
