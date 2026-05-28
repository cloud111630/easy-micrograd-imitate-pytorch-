"""
demo.py —— 测试 micrograd 引擎 & 与 PyTorch 对比

测试清单:
  1. 基本算术 (+, -, *, /, **, __rsub__) 的梯度正确性
  2. 激活函数 (tanh, relu, sigmoid, exp, log) 的梯度正确性
  3. 复合计算图自动求导 vs PyTorch autograd
  4. Neuron / Layer / MLP / Linear 前向 + 反向
  5. BCE 训练循环 (二分类, 正负混合数据)
  6. 数值梯度验证
"""
import math
import random
import sys

import torch

from micrograd import Linear, MLP, Module, Neuron, Value

# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────


def compare(label, ours, pytorch_val, tol=1e-6):
    """比较 micrograd 和 PyTorch 的值"""
    diff = abs(ours - pytorch_val)
    status = "PASS" if diff < tol else "FAIL"
    if status == "FAIL":
        print(
            f"  {status} {label}: ours={ours:.8f}  torch={pytorch_val:.8f}  diff={diff:.2e}"
        )
    else:
        print(f"  {status} {label}: {ours:.8f}")


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


# ──────────────────────────────────────────────
# Test 1: 基本算术 (含 __rsub__)
# ──────────────────────────────────────────────


def test_arithmetic():
    print("=" * 60)
    print("Test 1: 基本算术运算 (含 __rsub__)")
    print("=" * 60)

    # --- micrograd ---
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)

    d = a * b       # -6
    e = d + c       # 4
    f = Value(-2.0)
    L = e * f       # -8
    L.backward()

    # --- PyTorch ---
    a_t = torch.tensor(2.0, requires_grad=True)
    b_t = torch.tensor(-3.0, requires_grad=True)
    c_t = torch.tensor(10.0, requires_grad=True)
    f_t = torch.tensor(-2.0, requires_grad=True)

    d_t = a_t * b_t
    e_t = d_t + c_t
    L_t = e_t * f_t
    L_t.backward()

    print("前向传播值:")
    compare("d = a*b", d.data, d_t.item())
    compare("e = d+c", e.data, e_t.item())
    compare("L = e*f", L.data, L_t.item())

    print("反向传播梯度:")
    compare("a.grad", a.grad, a_t.grad.item())
    compare("b.grad", b.grad, b_t.grad.item())
    compare("c.grad", c.grad, c_t.grad.item())
    compare("f.grad", f.grad, f_t.grad.item())

    # --- __rsub__: 1 - Value ---
    v = Value(3.0)
    r = 1 - v       # 即 1 + (-3) = -2
    r.backward()

    v_t = torch.tensor(3.0, requires_grad=True)
    r_t = 1 - v_t
    r_t.backward()

    print("__rsub__ (1 - Value):")
    compare("  forward (1-3)", r.data, r_t.item())
    compare("  backward", v.grad, v_t.grad.item())

    # --- 除法 ---
    p = Value(6.0)
    q = Value(2.0)
    div = p / q
    div.backward()

    p_t = torch.tensor(6.0, requires_grad=True)
    q_t = torch.tensor(2.0, requires_grad=True)
    div_t = p_t / q_t
    div_t.backward()

    print("除法 (6 / 2):")
    compare("  forward", div.data, div_t.item())
    compare("  p.grad", p.grad, p_t.grad.item())
    compare("  q.grad", q.grad, q_t.grad.item())
    print()


# ──────────────────────────────────────────────
# Test 2: 激活函数 (含 log)
# ──────────────────────────────────────────────


def test_activations():
    print("=" * 60)
    print("Test 2: 激活函数 (tanh/relu/sigmoid/exp/log)")
    print("=" * 60)

    # --- tanh ---
    x = Value(0.5)
    o = x.tanh()
    o.backward()
    x_t = torch.tensor(0.5, requires_grad=True)
    o_t = torch.tanh(x_t)
    o_t.backward()
    print("tanh:")
    compare("  forward", o.data, o_t.item())
    compare("  backward", x.grad, x_t.grad.item())

    # --- relu ---
    x2 = Value(-0.5)
    o2 = x2.relu()
    o2.backward()
    x2_t = torch.tensor(-0.5, requires_grad=True)
    o2_t = torch.relu(x2_t)
    o2_t.backward()
    print("relu (x=-0.5):")
    compare("  forward", o2.data, o2_t.item())
    compare("  backward", x2.grad, x2_t.grad.item())

    x3 = Value(3.0)
    o3 = x3.relu()
    o3.backward()
    x3_t = torch.tensor(3.0, requires_grad=True)
    o3_t = torch.relu(x3_t)
    o3_t.backward()
    print("relu (x=3.0):")
    compare("  forward", o3.data, o3_t.item())
    compare("  backward", x3.grad, x3_t.grad.item())

    # --- sigmoid ---
    x4 = Value(0.0)
    o4 = x4.sigmoid()
    o4.backward()
    x4_t = torch.tensor(0.0, requires_grad=True)
    o4_t = torch.sigmoid(x4_t)
    o4_t.backward()
    print("sigmoid:")
    compare("  forward", o4.data, o4_t.item())
    compare("  backward", x4.grad, x4_t.grad.item())

    # --- exp ---
    x5 = Value(1.0)
    o5 = x5.exp()
    o5.backward()
    x5_t = torch.tensor(1.0, requires_grad=True)
    o5_t = torch.exp(x5_t)
    o5_t.backward()
    print("exp:")
    compare("  forward", o5.data, o5_t.item())
    compare("  backward", x5.grad, x5_t.grad.item())

    # --- log ---
    x6 = Value(0.5)
    o6 = x6.log()
    o6.backward()
    x6_t = torch.tensor(0.5, requires_grad=True)
    o6_t = torch.log(x6_t)
    o6_t.backward()
    print("log:")
    compare("  forward", o6.data, o6_t.item())
    compare("  backward", x6.grad, x6_t.grad.item())
    print()


# ──────────────────────────────────────────────
# Test 3: 复合计算图
# ──────────────────────────────────────────────


def test_composite_graph():
    print("=" * 60)
    print("Test 3: 复合计算图 (sigmoid / BCE 单条样本)")
    print("=" * 60)

    # micrograd: BCE(p=0.7, target=1)
    v = Value(0.7)
    tgt = 1.0
    out = -(tgt * v.log() + (1 - tgt) * (1 - v).log())
    out.backward()

    # PyTorch
    v_t = torch.tensor(0.7, requires_grad=True)
    tgt_t = torch.tensor(1.0)
    out_t = -(tgt_t * torch.log(v_t) + (1 - tgt_t) * torch.log(1 - v_t))
    out_t.backward()

    print("BCE 单样本 (p=0.7, tgt=1):")
    compare("  forward", out.data, out_t.item())
    compare("  backward", v.grad, v_t.grad.item())
    print()


# ──────────────────────────────────────────────
# Test 4: 共享节点 & 拓扑去重
# ──────────────────────────────────────────────


def test_shared_node():
    print("=" * 60)
    print("Test 4: 共享节点 a + a（测试梯度累加 & 拓扑去重）")
    print("=" * 60)

    a = Value(3.0)
    b = a + a
    b.backward()

    a_t = torch.tensor(3.0, requires_grad=True)
    b_t = a_t + a_t
    b_t.backward()

    print("a 被用了两次，b = a + a:")
    compare("  b.data", b.data, b_t.item())
    compare("  a.grad (应为 2.0)", a.grad, a_t.grad.item())
    print()


# ──────────────────────────────────────────────
# Test 5: Neuron / Layer / MLP / Linear
# ──────────────────────────────────────────────


def test_modules():
    print("=" * 60)
    print("Test 5: Neuron / Layer / MLP / Linear 模块")
    print("=" * 60)

    set_seed(42)
    x = [1.0, -2.0, 0.5]

    # --- Neuron with relu ---
    set_seed(42)
    neuron = Neuron(3, activation="relu")
    out = neuron(x)
    out.backward()

    # PyTorch 对标
    lin = torch.nn.Linear(3, 1)
    with torch.no_grad():
        for i, w in enumerate(neuron.w):
            lin.weight[0, i] = w.data
        lin.bias[0] = neuron.b.data
    x_t = torch.tensor(x, dtype=torch.float32)
    out_t = torch.relu(lin(x_t))
    out_t.backward()

    print("Neuron (relu) 前向:")
    compare("  output", out.data, out_t.item())
    print("Neuron (relu) 反向:")
    compare("  w[0].grad", neuron.w[0].grad, lin.weight.grad[0, 0].item())

    # --- MLP (relu hidden + sigmoid output) ---
    set_seed(42)
    mlp = MLP(3, [4, 4, 1])
    y = mlp(x)
    y.backward()

    # PyTorch 对标
    set_seed(42)
    pt_mlp = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.ReLU(),
        torch.nn.Linear(4, 4), torch.nn.ReLU(),
        torch.nn.Linear(4, 1), torch.nn.Sigmoid(),
    )
    with torch.no_grad():
        for i in range(3):
            layer = mlp.layers[i]
            pt_lin = pt_mlp[i * 2]
            for j, neuron in enumerate(layer.neurons):
                for k, w in enumerate(neuron.w):
                    pt_lin.weight[j, k] = w.data
                pt_lin.bias[j] = neuron.b.data
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = pt_mlp(x_t)
    y_t.backward()

    print("\nMLP (relu/sigmoid) 前向:")
    compare("  output", y.data, y_t.item())
    print("MLP 反向 (第1层第1个神经元 w[0]):")
    compare(
        "  w[0].grad",
        mlp.layers[0].neurons[0].w[0].grad,
        pt_mlp[0].weight.grad[0, 0].item(),
    )

    # --- Linear ---
    set_seed(42)
    linear = Linear(3, 2)
    out_lin = linear(x)
    for o in out_lin:
        o.backward()

    print("\nLinear (3→2) 前向:")
    print(f"  output[0] = {out_lin[0].data:.6f}")
    print(f"  output[1] = {out_lin[1].data:.6f}")

    # PyTorch 对标
    set_seed(42)
    pt_lin = torch.nn.Linear(3, 2)
    with torch.no_grad():
        for j in range(2):
            for k in range(3):
                pt_lin.weight[j, k] = linear.W[j][k].data
            pt_lin.bias[j] = linear.b[j].data
    x_t = torch.tensor(x, dtype=torch.float32)
    out_pt = pt_lin(x_t)
    out_pt.sum().backward()

    print("Linear 反向:")
    compare("  W[0][0].grad", linear.W[0][0].grad, pt_lin.weight.grad[0, 0].item())
    print()


# ──────────────────────────────────────────────
# Test 6: BCE 训练循环 (二分类)
# ──────────────────────────────────────────────


def test_training_loop():
    print("=" * 60)
    print("Test 6: BCE 训练循环 (正负混合数据)")
    print("=" * 60)

    set_seed(42)

    # 8 个样本，正负数混合
    xs = [
        [ 2.0,  3.0],
        [-3.0, -2.0],
        [ 0.5, -1.0],
        [-1.0,  0.5],
        [ 1.0,  1.0],
        [-2.0,  1.0],
        [ 3.0, -2.0],
        [-1.5, -1.5],
    ]
    ys = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]

    model = MLP(2, [8, 8, 1])
    lr = 0.05
    epochs = 100

    print("样本:")
    for i, (x, y) in enumerate(zip(xs, ys)):
        print(f"  {i}: x={x[0]:+5.1f} {x[1]:+5.1f}  y={y:.0f}")

    print(f"\n训练 {epochs} 轮 (lr={lr}):")
    for k in range(epochs):
        model.zero_grad()          # 必须在循环内

        ypred = [model(x) for x in xs]
        loss = sum(-(ygt * yout.log() + (1 - ygt) * (1 - yout).log())
                   for ygt, yout in zip(ys, ypred))

        loss.backward()

        for p in model.parameters():
            p.data += -lr * p.grad

        if k % 25 == 0:
            preds_str = "  ".join(f"{model(x).data:+.4f}" for x in xs)
            print(f"  epoch {k:3d}: loss={loss.data:.6f}  preds=[{preds_str}]")

    print("\n最终结果:")
    correct = 0
    for i, x in enumerate(xs):
        pred = model(x)
        pred_label = 1 if pred.data > 0.5 else 0
        ok = pred_label == ys[i]
        correct += ok
        print(
            f"  样本{i}: pred={pred.data:+.4f} -> {pred_label}  "
            f"target={ys[i]:.0f}  {'PASS' if ok else 'FAIL'}"
        )
    print(f"\n准确率: {correct}/{len(xs)}  ({100*correct/len(xs):.1f}%)")
    print()


# ──────────────────────────────────────────────
# Test 7: 数值梯度验证
# ──────────────────────────────────────────────


def test_numerical_gradient():
    print("=" * 60)
    print("Test 7: 数值梯度验证 (finite difference)")
    print("=" * 60)

    def numerical_grad(f, x, h=1e-6):
        return (f(x + h) - f(x - h)) / (2 * h)

    checks = [
        ("sigmoid", 0.3,  lambda v: v.sigmoid(), lambda x: 1/(1+math.exp(-x))),
        ("log",     0.5,  lambda v: v.log(),     math.log),
        ("relu+",   2.0,  lambda v: v.relu(),    lambda x: max(0, x)),
        ("relu-",  -1.0,  lambda v: v.relu(),    lambda x: max(0, x)),
    ]

    for name, x_val, micrograd_fn, numpy_fn in checks:
        v = Value(x_val)
        out = micrograd_fn(v)
        out.backward()
        num = numerical_grad(numpy_fn, x_val)
        diff = abs(v.grad - num)
        print(f"  {name:8s}: auto={v.grad:.8f}  num={num:.8f}  diff={diff:.2e}")

    print()


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────


def main():
    print("\n" + "=" * 60)
    print("  micrograd vs PyTorch —— 完整验证")
    print("=" * 60 + "\n")

    test_arithmetic()
    test_activations()
    test_composite_graph()
    test_shared_node()
    test_modules()
    test_training_loop()
    test_numerical_gradient()

    print("=" * 60)
    print("  全部测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
