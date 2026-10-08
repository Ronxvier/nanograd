import random

import pytest

from src.engine import Value
from src.nn import Neuron, Layer, MLP


@pytest.fixture(autouse=True)
def seed():
    random.seed(0)


def test_neuron_parameters():
    n = Neuron(3)
    params = n.parameters()
    assert len(params) == 4  # 3 weights + bias
    assert all(isinstance(p, Value) for p in params)
    assert all(-1 <= p.data <= 1 for p in params)

def test_neuron_forward_matches_manual():
    n = Neuron(2)
    x = [0.5, -1.5]
    expected = Value(n.w[0].data * x[0] + n.w[1].data * x[1] + n.b.data).tanh().data
    out = n(x)
    assert isinstance(out, Value)
    assert out.data == pytest.approx(expected)
    assert -1 < out.data < 1

def test_neuron_backward_reaches_parameters():
    n = Neuron(3)
    n([1.0, 2.0, -1.0]).backward()
    assert all(p.grad != 0 for p in n.parameters())

def test_layer_shapes():
    layer = Layer(3, 4)
    assert len(layer.neurons) == 4
    assert len(layer.parameters()) == 4 * (3 + 1)
    out = layer([1.0, 2.0, 3.0])
    assert isinstance(out, list) and len(out) == 4

def test_layer_single_output_is_unwrapped():
    out = Layer(3, 1)([1.0, 2.0, 3.0])
    assert isinstance(out, Value)

def test_mlp_structure():
    model = MLP(3, [4, 4, 1])
    assert len(model.layers) == 3
    assert [len(l.neurons) for l in model.layers] == [4, 4, 1]
    assert [len(l.neurons[0].w) for l in model.layers] == [3, 4, 4]
    assert len(model.parameters()) == (3 + 1) * 4 + (4 + 1) * 4 + (4 + 1) * 1

def test_mlp_parameters_are_unique():
    model = MLP(2, [3, 2])
    params = model.parameters()
    assert len({id(p) for p in params}) == len(params)

def test_mlp_forward_output():
    out = MLP(3, [4, 4, 1])([2.0, 3.0, -1.0])
    assert isinstance(out, Value)
    assert -1 < out.data < 1

def test_mlp_multiple_outputs():
    out = MLP(3, [4, 2])([2.0, 3.0, -1.0])
    assert isinstance(out, list) and len(out) == 2

def test_mlp_accepts_value_inputs():
    x = [Value(1.0), Value(-2.0)]
    out = MLP(2, [3, 1])(x)
    out.backward()
    # gradient should flow back to the inputs too
    assert any(xi.grad != 0 for xi in x)

def test_mlp_gradient_matches_finite_difference():
    model = MLP(2, [3, 1])
    x = [0.7, -0.4]
    out = model(x)
    out.backward()
    h = 1e-6
    for p in model.parameters():
        orig = p.data
        p.data = orig + h
        plus = model(x).data
        p.data = orig - h
        minus = model(x).data
        p.data = orig
        assert p.grad == pytest.approx((plus - minus) / (2 * h), abs=1e-5)

def test_single_gradient_step_reduces_loss():
    model = MLP(3, [4, 1])
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5]]
    ys = [1.0, -1.0]

    def loss_fn():
        return sum((model(x) - y) ** 2 for x, y in zip(xs, ys))

    loss = loss_fn()
    loss.backward()
    for p in model.parameters():
        p.data -= 0.01 * p.grad
    assert loss_fn().data < loss.data
