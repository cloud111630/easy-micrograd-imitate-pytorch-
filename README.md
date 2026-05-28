# micrograd

从零实现的自动求导引擎 + 多层感知机，纯 Python 编写，**仅依赖 `math` 库**。

对标 PyTorch 的 `Tensor` / `autograd` / `nn.Module`，所有梯度计算均通过 `demo.py` 与 PyTorch 逐项对比验证。

## 安装

```bash
# 本地开发模式
pip install -e .

# 或直接从 GitHub 安装
pip install git+https://github.com/cloud111630/easy-micrograd-imitate-pytorch-.git
```

## 项目结构

```
micrograd/
  __init__.py        # 对外暴露 Value, Module, Neuron, Layer, MLP, Linear
  engine.py          # Value 类 — 自动求导引擎
  nn.py              # Module / Neuron / Layer / MLP / Linear — 神经网络模块
demo.py              # 7 组测试，每项与 PyTorch 对照
pyproject.toml       # pip 安装配置
```

## 快速开始

```python
from micrograd import Value, MLP

# ---- 1. 自动求导 ----
a = Value(2.0)
b = Value(-3.0)
c = a * b + Value(10.0)   # c = 2*(-3) + 10 = 4
c.backward()
print(a.grad)              # 6.0
print(b.grad)              # -4.0

# ---- 2. 二分类训练 (8 样本, BCE loss) ----
xs = [[2,3], [-3,-2], [0.5,-1], [-1,0.5], [1,1], [-2,1], [3,-2], [-1.5,-1.5]]
ys = [1, 0, 0, 1, 1, 1, 0, 0]

model = MLP(2, [8, 8, 1])  # 2输入 → 8 → 8 → 1(sigmoid)

for epoch in range(100):
    model.zero_grad()
    ypred = [model(x) for x in xs]
    loss = sum(-(ygt * p.log() + (1-ygt) * (1-p).log())
               for ygt, p in zip(ys, ypred))
    loss.backward()
    for p in model.parameters():
        p.data += -0.05 * p.grad

# 预测
for x in xs:
    pred = model(x)
    print(f"x={x}  pred={pred.data:.4f}  label={1 if pred.data>0.5 else 0}")
```

## API

### `Value` — 对标 `torch.Tensor`

```python
class Value(data, _children=(), _op='', label='')
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `data` | `float / int` | 标量数值 |
| `_children` | `tuple[Value]` | 上游节点，内部使用 |
| `_op` | `str` | 运算符号，仅用于可视化 |
| `label` | `str` | 标签，仅用于可视化 |

**算术运算符**：`+` `-` `*` `/` `**` （含 `__radd__` `__rmul__` `__rsub__`，支持 `3 + Value(2.0)` 等混合运算）

**激活函数**：`.tanh()` `.relu()` `.sigmoid()` `.exp()` `.log()`

**反向传播**：`.backward()` — 自动拓扑排序 + 链式法则

**梯度清零**：`.zero_grad()` — 递归清零

---

### `Module` — 对标 `torch.nn.Module`

```python
class Module:
    def zero_grad(self)   # 清零所有参数梯度
    def parameters(self)  # 返回参数列表，子类覆写
```

### `Neuron(Module)` — 单个神经元

```python
class Neuron(nin, activation='relu')
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `nin` | `int` | 输入维度 |
| `activation` | `str` | `'relu'` / `'tanh'` / `'sigmoid'` / `'linear'` |

### `Layer(Module)` — 全连接层

```python
class Layer(nin, nout, activation='relu')
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `nin` | `int` | 输入维度 |
| `nout` | `int` | 本层神经元数 |
| `activation` | `str` | 透传给每个 Neuron |

### `MLP(Module)` — 多层感知机

```python
class MLP(nin, nouts)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `nin` | `int` | 输入维度 |
| `nouts` | `list[int]` | 每层神经元数，如 `[8, 8, 1]` |

隐藏层自动使用 ReLU，输出层自动使用 Sigmoid。

### `Linear(Module)` — 线性变换层

```python
class Linear(nin, nout)
```

纯 `W·x + b`，不含激活函数，对标 `torch.nn.Linear`。

---

## 测试

```bash
python demo.py
```

7 组测试，**每个测试用相同数值同时跑 micrograd 和 PyTorch，逐项对比**：

| 测试 | 内容 | 验证方式 |
|------|------|---------|
| 1 | 算术运算 (+, -, *, /, **, __rsub__) | vs `torch.Tensor` |
| 2 | 激活函数 (tanh, relu, sigmoid, exp, log) 前向 + 反向 | vs `torch` |
| 3 | BCE 损失单样本梯度 | vs `torch` |
| 4 | 共享节点梯度累加 (a + a) | vs `torch` |
| 5 | Neuron / Layer / MLP / Linear 前向 + 反向 | vs `torch.nn` |
| 6 | BCE 训练循环 (8 样本, 100 epoch) | 准确率 |
| 7 | 数值梯度 (有限差分) vs autograd | 误差 < 1e-10 |

## License

MIT
