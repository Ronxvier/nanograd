import math

import pytest

from src.engine import Value


def numerical_grad(f, xs, i, h=1e-6):
    # Central difference of f with respect to xs[i], f takes plain floats
    plus = list(xs)
    minus = list(xs)
    plus[i] += h
    minus[i] -= h
    return (f(*plus) - f(*minus)) / (2 * h)


def check_grads(f, xs, tol=1e-5):
    # Run f on Values, backprop, and compare every input's grad against finite differences
    vals = [Value(x) for x in xs]
    out = f(*vals)
    out.backward()
    for i, v in enumerate(vals):
        expected = numerical_grad(lambda *a: f(*[Value(x) for x in a]).data, xs, i)
        assert v.grad == pytest.approx(expected, abs=tol), f"grad mismatch for input {i}"


# Forward pass

def test_add():
    assert (Value(2.0) + Value(3.0)).data == 5.0

def test_mul():
    assert (Value(2.0) * Value(-3.0)).data == -6.0

def test_sub():
    assert (Value(2.0) - Value(5.0)).data == -3.0

def test_neg():
    assert (-Value(4.0)).data == -4.0

def test_div():
    assert (Value(6.0) / Value(4.0)).data == pytest.approx(1.5)

def test_pow():
    assert (Value(3.0) ** 2).data == 9.0
    assert (Value(4.0) ** 0.5).data == pytest.approx(2.0)
    assert (Value(2.0) ** -1).data == pytest.approx(0.5)

def test_exp():
    assert Value(1.0).exp().data == pytest.approx(math.e)

@pytest.mark.parametrize("x", [-3.0, -0.5, 0.0, 0.5, 3.0])
def test_tanh_matches_math(x):
    assert Value(x).tanh().data == pytest.approx(math.tanh(x))

def test_ops_with_python_numbers():
    a = Value(3.0)
    assert (a + 1).data == 4.0
    assert (1 + a).data == 4.0
    assert (a * 2).data == 6.0
    assert (2 * a).data == 6.0
    assert (a - 1).data == 2.0
    assert (10 - a).data == 7.0
    assert (a / 2).data == pytest.approx(1.5)
    assert (6 / a).data == pytest.approx(2.0)

def test_repr():
    assert repr(Value(1.5)) == "Value(data=1.5)"

def test_graph_bookkeeping():
    a, b = Value(1.0), Value(2.0)
    c = a + b
    d = a * b
    assert c._prev == {a, b} and c._op == '+'
    assert d._prev == {a, b} and d._op == '*'
    assert Value(1.0)._prev == set()
    assert Value(1.0).grad == 0.0


# Backward pass: individual ops

def test_add_backward():
    check_grads(lambda a, b: a + b, [2.0, -3.0])

def test_mul_backward():
    check_grads(lambda a, b: a * b, [2.0, -3.0])

def test_sub_backward():
    check_grads(lambda a, b: a - b, [2.0, -3.0])

def test_div_backward():
    check_grads(lambda a, b: a / b, [2.0, -3.0])

def test_neg_backward():
    check_grads(lambda a: -a, [2.0])

@pytest.mark.parametrize("p", [2, 3, 0.5, -1, -2])
def test_pow_backward(p):
    check_grads(lambda a: a ** p, [1.7])

def test_exp_backward():
    check_grads(lambda a: a.exp(), [0.7])

@pytest.mark.parametrize("x", [-2.0, -0.3, 0.0, 0.3, 2.0])
def test_tanh_backward(x):
    check_grads(lambda a: a.tanh(), [x])

def test_rops_backward():
    check_grads(lambda a: 2 * a + 1, [1.5])
    check_grads(lambda a: 1 - a, [1.5])
    check_grads(lambda a: 3 / a, [1.5])


# Backward pass: graph structure

def test_reused_node_accumulates_grad():
    # a is used twice, so its gradient must accumulate rather than be overwritten
    a = Value(3.0)
    b = a + a
    b.backward()
    assert a.grad == 2.0

def test_reused_node_in_mul():
    a = Value(3.0)
    b = a * a
    b.backward()
    assert a.grad == 6.0

def test_diamond_graph():
    # a feeds two branches that later merge
    check_grads(lambda a: (a * 2) * (a + 1), [1.5])

def test_backward_sets_output_grad_to_one():
    a = Value(2.0)
    out = a * 3
    out.backward()
    assert out.grad == 1.0

def test_karpathy_neuron_example():
    # Example from the README / micrograd lecture
    x1, x2 = Value(2.0), Value(0.0)
    w1, w2 = Value(-3.0), Value(1.0)
    b = Value(6.8813735870195432)
    o = (x1 * w1 + x2 * w2 + b).tanh()
    o.backward()
    assert o.data == pytest.approx(0.7071, abs=1e-4)
    assert x1.grad == pytest.approx(-1.5, abs=1e-4)
    assert w1.grad == pytest.approx(1.0, abs=1e-4)
    assert x2.grad == pytest.approx(0.5, abs=1e-4)
    assert w2.grad == pytest.approx(0.0, abs=1e-4)

def test_tanh_via_exp_matches_tanh():
    # Building tanh out of primitives should give the same value and gradient as the fused op
    def manual(a):
        e = (2 * a).exp()
        return (e - 1) / (e + 1)
    a1, a2 = Value(0.8), Value(0.8)
    o1, o2 = a1.tanh(), manual(a2)
    o1.backward()
    o2.backward()
    assert o1.data == pytest.approx(o2.data)
    assert a1.grad == pytest.approx(a2.grad)

def test_complex_expression():
    def f(a, b, c):
        d = a * b + c ** 2
        e = (d / (a + 3)).tanh()
        return e * b.exp() - c
    check_grads(f, [0.5, -1.2, 0.8])

def test_deep_chain():
    # Long chain to make sure topological sort handles depth
    def f(a):
        x = a
        for _ in range(50):
            x = x * 0.99 + 0.01
        return x
    check_grads(f, [1.3])


# Known gaps

@pytest.mark.xfail(reason="tanh computes exp(2x) directly and overflows for large x", raises=OverflowError, strict=True)
def test_tanh_large_input():
    assert Value(400.0).tanh().data == pytest.approx(1.0)

@pytest.mark.xfail(reason="__pow__ does not backprop into a Value exponent (TODO in engine.py)", strict=True)
def test_pow_backward_into_exponent():
    a, p = Value(2.0), Value(3.0)
    (a ** p).backward()
    assert p.grad == pytest.approx(8.0 * math.log(2.0))
