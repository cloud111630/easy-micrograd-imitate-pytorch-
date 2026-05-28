import random
from .engine import Value


# ---------- Module 基类 ----------

class Module:
    """所有神经网络模块的基类"""

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


# ---------- Neuron ----------

class Neuron(Module):
    """单个神经元: activation(w·x + b)"""

    def __init__(self, nin, activation="relu"):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == "relu":
            return act.relu()
        elif self.activation == "tanh":
            return act.tanh()
        elif self.activation == "sigmoid":
            return act.sigmoid()
        else:
            return act  # linear: 不做激活

    def parameters(self):
        return self.w + [self.b]

    def set_weights(self, weights, bias):
        """手动设置权重（用于和 PyTorch 对齐对比）"""
        for w, val in zip(self.w, weights):
            w.data = val
        self.b.data = bias


# ---------- Layer ----------

class Layer(Module):
    """全连接层: nout 个并列的 Neuron"""

    def __init__(self, nin, nout, activation="relu"):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


# ---------- MLP ----------

class MLP(Module):
    """多层感知机: 隐藏层 ReLU / 输出层 Sigmoid"""

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            is_last = (i == len(nouts) - 1)
            act = "sigmoid" if is_last else "relu"
            self.layers.append(Layer(sz[i], sz[i + 1], activation=act))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ---------- Linear (独立的线性层, 像 PyTorch) ----------

class Linear(Module):
    """线性变换层: W·x + b, 不含激活"""

    def __init__(self, nin, nout):
        self.W = [[Value(random.uniform(-1, 1)) for _ in range(nin)] for _ in range(nout)]
        self.b = [Value(random.uniform(-1, 1)) for _ in range(nout)]

    def __call__(self, x):
        return [sum((w * xi for w, xi in zip(row, x)), b) for row, b in zip(self.W, self.b)]

    def parameters(self):
        return [p for row in self.W for p in row] + self.b
