from src.engine import *
import numpy as np
import pytest

def test_TensorOps():
    t = Tensor([[[1,2],[1,2]],[[1,2],[1,2]],[[1,2],[1,2]],[[1,2],[1,2]]])
    p = t
    t = t*p
    print(t.shape)
    print(t)
    assert t.shape == (4, 2, 2), "Incorrect Shape"
    for i in np.ndindex(t.shape):
        val = t.values[i]
        assert isinstance(val, Value), "Non-value object found in Tensor"


def data(t):
    # Plain float array of a Tensor's values, for comparing against numpy
    return np.vectorize(lambda v: v.data, otypes=[float])(t.values)

def test_construction_wraps_scalars():
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t.shape == (2, 2)
    assert all(isinstance(v, Value) for v in t.values.flat)
    np.testing.assert_allclose(data(t), [[1, 2], [3, 4]])

def test_construction_keeps_existing_values():
    a = Value(5.0)
    t = Tensor([a, 1.0])
    assert t.values[0] is a

def test_construction_1d_and_3d():
    assert Tensor([1, 2, 3]).shape == (3,)
    assert Tensor(np.zeros((2, 3, 4))).shape == (2, 3, 4)

def test_elementwise_ops():
    a = np.array([[1.0, -2.0], [3.0, 0.5]])
    b = np.array([[4.0, 5.0], [-1.0, 2.0]])
    ta, tb = Tensor(a), Tensor(b)
    np.testing.assert_allclose(data(ta + tb), a + b)
    np.testing.assert_allclose(data(ta - tb), a - b)
    np.testing.assert_allclose(data(ta * tb), a * b)

@pytest.mark.parametrize("op", ["__add__", "__sub__", "__mul__"])
def test_elementwise_shape_mismatch(op):
    with pytest.raises(AssertionError):
        getattr(Tensor([[1, 2]]), op)(Tensor([[1], [2]]))

def test_matmul_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 4))
    b = rng.normal(size=(4, 2))
    out = Tensor(a) @ Tensor(b)
    assert out.shape == (3, 2)
    np.testing.assert_allclose(data(out), a @ b)

def test_matmul_dimension_mismatch():
    with pytest.raises(AssertionError):
        Tensor(np.ones((2, 3))) @ Tensor(np.ones((2, 3)))

def test_matmul_requires_2d():
    with pytest.raises(AssertionError):
        Tensor([1, 2]) @ Tensor([[1], [2]])

def test_tanh():
    a = np.array([[-2.0, 0.0], [0.5, 3.0]])
    np.testing.assert_allclose(data(Tensor(a).tanh()), np.tanh(a))

def test_sum_and_mean():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t = Tensor(a)
    s, m = t.sum(), t.mean()
    assert isinstance(s, Value) and isinstance(m, Value)
    assert s.data == pytest.approx(21.0)
    assert m.data == pytest.approx(3.5)

def test_sum_backward():
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    t.sum().backward()
    np.testing.assert_allclose(np.vectorize(lambda v: v.grad)(t.values), np.ones((2, 2)))

def test_mean_backward():
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    t.mean().backward()
    np.testing.assert_allclose(np.vectorize(lambda v: v.grad)(t.values), np.full((2, 2), 0.25))

def test_ops_share_graph_with_inputs():
    # Results must be built from the input Values so gradients can flow back
    a, b = Tensor([[1.0, 2.0]]), Tensor([[3.0, 4.0]])
    (a * b).sum().backward()
    np.testing.assert_allclose(np.vectorize(lambda v: v.grad)(a.values), [[3.0, 4.0]])
    np.testing.assert_allclose(np.vectorize(lambda v: v.grad)(b.values), [[1.0, 2.0]])

def test_matmul_backward():
    # For L = sum(tanh(A @ B)): dL/dA = G @ B.T, dL/dB = A.T @ G, where G = 1 - tanh(A @ B)^2
    rng = np.random.default_rng(1)
    a = rng.normal(size=(2, 3))
    b = rng.normal(size=(3, 2))
    ta, tb = Tensor(a), Tensor(b)
    (ta @ tb).tanh().sum().backward()
    g = 1 - np.tanh(a @ b) ** 2
    grad = np.vectorize(lambda v: v.grad, otypes=[float])
    np.testing.assert_allclose(grad(ta.values), g @ b.T)
    np.testing.assert_allclose(grad(tb.values), a.T @ g)
