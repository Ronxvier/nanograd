import numpy as np
import pytest

from src.Tensor import Tensor


def numerical_grad(f, xs, i, h=1e-6):
    # Central difference of sum(f) with respect to every element of xs[i], f takes plain arrays
    grad = np.zeros_like(xs[i])
    for idx in np.ndindex(xs[i].shape):
        plus = [x.copy() for x in xs]
        minus = [x.copy() for x in xs]
        plus[i][idx] += h
        minus[i][idx] -= h
        grad[idx] = (f(*plus) - f(*minus)) / (2 * h)
    return grad


def check_grads(f, xs, tol=1e-5):
    # Run f on Tensors, backprop, and compare every input's grads against finite differences.
    # backward() seeds the output with ones, so the reference is the gradient of sum(f)
    xs = [np.asarray(x, dtype=np.float64) for x in xs]
    tensors = [Tensor(x.copy()) for x in xs]
    f(*tensors).backward()
    for i, t in enumerate(tensors):
        expected = numerical_grad(lambda *a: f(*[Tensor(x) for x in a]).data.sum(), xs, i)
        assert t.grads.shape == xs[i].shape, f"grads shape mismatch for input {i}"
        np.testing.assert_allclose(t.grads, expected, atol=tol, err_msg=f"grads mismatch for input {i}")


A = [[1.0, -2.0, 3.0], [0.5, 4.0, -1.5]]
B = [[2.0, 0.5, -1.0], [3.0, -0.25, 1.7]]
POS = [[1.0, 2.0, 3.0], [0.5, 4.0, 1.5]]
ROW = [10.0, 20.0, 30.0]       # broadcasts against (2, 3) along axis 0
COL = [[2.0], [3.0]]           # broadcasts against (2, 3) along axis 1
POS_ROW = [1.5, 2.0, 0.5]


# Construction

def test_data_is_float64_array():
    t = Tensor([1, 2, 3])
    assert isinstance(t.data, np.ndarray)
    assert t.data.dtype == np.float64

def test_grads_start_as_zeros_in_shape_of_data():
    t = Tensor(A)
    assert t.grads.shape == (2, 3)
    assert np.all(t.grads == 0)

def test_scalar_tensor():
    t = Tensor(3.0)
    assert t.data.shape == ()
    assert t.grads.shape == ()

def test_leaf_has_no_children():
    assert Tensor(A)._prev == set()

def test_repr():
    assert repr(Tensor([1.0, 2.0])) == "Tensor(data=[1. 2.])"

def test_ops_record_children_and_op():
    a, b = Tensor(A), Tensor(B)
    out = a + b
    assert out._prev == {a, b}
    assert out._op == '+'
    assert (a * b)._op == '*'
    assert (a @ Tensor(np.ones((3, 2))))._op == '@'
    assert (a ** 2)._op == '**'
    assert (-a)._op == 'neg'
    assert a.exp()._op == 'exp'
    assert a.relu()._op == 'relu'
    assert a.tanh()._op == 'tanh'
    assert a.sum()._op == 'sum'


# Forward pass

def test_add():
    np.testing.assert_array_equal((Tensor(A) + Tensor(B)).data, np.array(A) + np.array(B))

def test_mul():
    np.testing.assert_array_equal((Tensor(A) * Tensor(B)).data, np.array(A) * np.array(B))

def test_sub():
    np.testing.assert_array_equal((Tensor(A) - Tensor(B)).data, np.array(A) - np.array(B))

def test_div():
    np.testing.assert_allclose((Tensor(A) / Tensor(B)).data, np.array(A) / np.array(B))

def test_neg():
    np.testing.assert_array_equal((-Tensor(A)).data, -np.array(A))

def test_pow_scalar_exponent():
    np.testing.assert_array_equal((Tensor(A) ** 2).data, np.array(A) ** 2)

def test_pow_tensor_exponent():
    np.testing.assert_allclose((Tensor(POS) ** Tensor(B)).data, np.array(POS) ** np.array(B))

def test_matmul():
    w = np.arange(6.0).reshape(3, 2)
    np.testing.assert_allclose((Tensor(A) @ Tensor(w)).data, np.array(A) @ w)

def test_exp():
    np.testing.assert_allclose(Tensor(A).exp().data, np.exp(np.array(A)))

def test_relu():
    np.testing.assert_array_equal(Tensor([-1.0, 0.0, 2.0]).relu().data, [0.0, 0.0, 2.0])

def test_tanh():
    np.testing.assert_allclose(Tensor(A).tanh().data, np.tanh(np.array(A)))

def test_tanh_large_inputs_saturate_instead_of_nan():
    np.testing.assert_array_equal(Tensor([400.0, -400.0]).tanh().data, [1.0, -1.0])

def test_sum():
    t = Tensor(A)
    assert t.sum().data == pytest.approx(np.sum(A))
    np.testing.assert_allclose(t.sum(axis=0).data, np.sum(A, axis=0))
    np.testing.assert_allclose(t.sum(axis=1).data, np.sum(A, axis=1))
    assert t.sum(axis=1, keepdims=True).data.shape == (2, 1)

def test_mean():
    t = Tensor(A)
    assert t.mean().data == pytest.approx(np.mean(A))
    np.testing.assert_allclose(t.mean(axis=0).data, np.mean(A, axis=0))
    np.testing.assert_allclose(t.mean(axis=1).data, np.mean(A, axis=1))
    assert t.mean(axis=1, keepdims=True).data.shape == (2, 1)

def test_ops_accept_raw_arrays():
    np.testing.assert_array_equal((Tensor(A) + B).data, np.array(A) + np.array(B))
    np.testing.assert_array_equal((Tensor(A) * B).data, np.array(A) * np.array(B))
    np.testing.assert_array_equal((Tensor(A) - B).data, np.array(A) - np.array(B))
    np.testing.assert_allclose((Tensor(A) @ np.ones((3, 2))).data, np.array(A) @ np.ones((3, 2)))


# Python numbers on either side

def test_scalar_on_right():
    np.testing.assert_array_equal((Tensor(A) + 1.0).data, np.array(A) + 1.0)
    np.testing.assert_array_equal((Tensor(A) - 1.0).data, np.array(A) - 1.0)
    np.testing.assert_array_equal((Tensor(A) * 2.0).data, np.array(A) * 2.0)
    np.testing.assert_allclose((Tensor(A) / 2.0).data, np.array(A) / 2.0)

def test_scalar_on_left():
    np.testing.assert_array_equal((1.0 + Tensor(A)).data, 1.0 + np.array(A))
    np.testing.assert_array_equal((1.0 - Tensor(A)).data, 1.0 - np.array(A))
    np.testing.assert_array_equal((2.0 * Tensor(A)).data, 2.0 * np.array(A))
    np.testing.assert_allclose((2.0 / Tensor(A)).data, 2.0 / np.array(A))

def test_scalar_tensor_on_left():
    np.testing.assert_array_equal((Tensor(2.0) + Tensor(A)).data, 2.0 + np.array(A))
    np.testing.assert_array_equal((Tensor(2.0) * Tensor(A)).data, 2.0 * np.array(A))


# Broadcasting

def test_broadcast_row():
    np.testing.assert_array_equal((Tensor(A) + Tensor(ROW)).data, np.array(A) + np.array(ROW))

def test_broadcast_column():
    np.testing.assert_array_equal((Tensor(A) * Tensor(COL)).data, np.array(A) * np.array(COL))

def test_broadcast_column_with_row():
    assert (Tensor(COL) + Tensor(ROW)).data.shape == (2, 3)

@pytest.mark.parametrize("op", [
    lambda a, b: a + b,
    lambda a, b: a * b,
    lambda a, b: a - b,
    lambda a, b: a ** b,
], ids=["add", "mul", "sub", "pow"])
def test_incompatible_shapes_raise(op):
    with pytest.raises(ValueError):
        op(Tensor([1.0, 2.0, 3.0]), Tensor([1.0, 2.0]))

def test_matmul_inner_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        Tensor(A) @ Tensor(A)


# Backward pass: element-wise ops

def test_add_grads():
    check_grads(lambda a, b: a + b, [A, B])

def test_mul_grads():
    check_grads(lambda a, b: a * b, [A, B])

def test_sub_grads():
    check_grads(lambda a, b: a - b, [A, B])

def test_div_grads():
    check_grads(lambda a, b: a / b, [A, B])

def test_neg_grads():
    check_grads(lambda a: -a, [A])

def test_pow_scalar_exponent_grads():
    check_grads(lambda a: a ** 3, [A])

def test_pow_negative_exponent_grads():
    check_grads(lambda a: a ** -1, [A])

def test_pow_tensor_exponent_grads():
    check_grads(lambda a, b: a ** b, [POS, B])

def test_pow_negative_base_leaves_exponent_grads_zero():
    # log of a negative base is undefined, so the exponent gets no gradient
    a, n = Tensor([-1.0, -2.0]), Tensor(2.0)
    (a ** n).backward()
    np.testing.assert_allclose(a.grads, [-2.0, -4.0])
    assert n.grads == 0.0

def test_exp_grads():
    check_grads(lambda a: a.exp(), [A])

def test_relu_grads():
    check_grads(lambda a: a.relu(), [A])

def test_relu_grads_at_zero_is_zero():
    a = Tensor([-1.0, 0.0, 2.0])
    a.relu().backward()
    np.testing.assert_array_equal(a.grads, [0.0, 0.0, 1.0])

def test_tanh_grads():
    check_grads(lambda a: a.tanh(), [A])


# Backward pass: Python numbers and scalar tensors

def test_python_numbers_on_either_side_grads():
    check_grads(lambda a: (2.0 + a) * (2.0 * a) + (2.0 - a) + (2.0 / a) + (a - 1.0) / 2.0, [A])

def test_scalar_tensor_on_right_grads():
    # the scalar's grads are summed over every element it was applied to
    check_grads(lambda a, s: a * s + a - s, [A, 2.5])

def test_scalar_tensor_on_left_grads():
    check_grads(lambda s, a: s * a + s - a + s / a, [2.5, A])

def test_scalar_tensor_base_grads():
    check_grads(lambda s, a: s ** a, [2.5, A])

def test_scalar_tensors_grads():
    check_grads(lambda a, b: a * b + a ** 2 - b / a, [3.0, -2.0])


# Backward pass: broadcasting

@pytest.mark.parametrize("op", [
    lambda a, b: a + b,
    lambda a, b: a * b,
    lambda a, b: a - b,
    lambda a, b: a / b,
], ids=["add", "mul", "sub", "div"])
@pytest.mark.parametrize("shape_pair", [
    (A, ROW), (ROW, A), (A, COL), (COL, A), (COL, ROW),
], ids=["matrix-row", "row-matrix", "matrix-col", "col-matrix", "col-row"])
def test_broadcast_grads(op, shape_pair):
    check_grads(op, list(shape_pair))

@pytest.mark.parametrize("shape_pair", [
    (POS, POS_ROW), (POS_ROW, A), (POS, COL), (COL, A),
], ids=["matrix-row", "row-matrix", "matrix-col", "col-matrix"])
def test_broadcast_pow_grads(shape_pair):
    check_grads(lambda a, b: a ** b, list(shape_pair))

def test_bias_grads_are_summed_over_batch():
    # the worked example: a (3,) bias added to two rows collects both rows' grads
    x, b = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), Tensor(ROW)
    ((x + b) * Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).backward()
    np.testing.assert_allclose(b.grads, [5.0, 7.0, 9.0])

def test_column_grads_are_summed_over_row():
    x, c = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), Tensor(COL)
    ((x * c) * Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).backward()
    np.testing.assert_allclose(c.grads, [[14.0], [77.0]])


# Backward pass: matmul

def test_matmul_grads():
    check_grads(lambda a, w: a @ w, [A, np.arange(6.0).reshape(3, 2)])

def test_matmul_worked_example():
    a, b = Tensor([[1.0, 2.0], [3.0, 4.0]]), Tensor([[5.0], [6.0]])
    out = a @ b
    (out * Tensor([[1.0], [2.0]])).backward()
    np.testing.assert_allclose(out.data, [[17.0], [39.0]])
    np.testing.assert_allclose(a.grads, [[5.0, 6.0], [10.0, 12.0]])
    np.testing.assert_allclose(b.grads, [[7.0], [10.0]])

def test_matmul_square_same_tensor_grads():
    check_grads(lambda a: a @ a, [[[1.0, -2.0, 0.5], [3.0, 0.25, -1.0], [2.0, 1.0, 4.0]]])

def test_matmul_single_row_and_column_grads():
    check_grads(lambda x, w: x @ w, [[[1.0, -2.0, 3.0]], np.arange(6.0).reshape(3, 2)])
    check_grads(lambda x, w: x @ w, [A, [[1.0], [-1.0], [0.5]]])


# Backward pass: reductions

@pytest.mark.parametrize("reduce", ["sum", "mean"])
@pytest.mark.parametrize("kwargs, weight", [
    ({}, None),
    ({"axis": 0}, [1.0, -2.0, 0.5]),
    ({"axis": 1}, [3.0, -1.0]),
    ({"axis": 1, "keepdims": True}, [[3.0], [-1.0]]),
    ({"axis": -1}, [3.0, -1.0]),
], ids=["all", "axis0", "axis1", "axis1-keepdims", "axis-1"])
def test_reduction_grads(reduce, kwargs, weight):
    # weight the reduced output so each output element gets a different gradient
    def f(a):
        out = getattr(a, reduce)(**kwargs)
        return out if weight is None else out * weight
    check_grads(f, [A])

@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_reduction_square_input_grads(reduce):
    # a square input is where putting the gradient back on the wrong axis would go unnoticed by shape alone
    square = [[1.0, -2.0, 0.5], [3.0, 0.25, -1.0], [2.0, 1.0, 4.0]]
    check_grads(lambda a: getattr(a, reduce)(axis=1) * [3.0, -1.0, 2.0], [square])

@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_reduction_tuple_axis_grads(reduce):
    x = np.arange(24.0).reshape(2, 3, 4) / 10
    check_grads(lambda a: getattr(a, reduce)(axis=(0, 2)) * [1.0, -2.0, 0.5], [x])

def test_sum_grads_worked_example():
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    (x.sum(axis=1) * Tensor([10.0, 20.0])).backward()
    np.testing.assert_allclose(x.grads, [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])

def test_mean_grads_are_one_over_n():
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x.mean().backward()
    np.testing.assert_allclose(x.grads, np.full((2, 3), 1 / 6))


# Backward pass: graphs

def test_same_tensor_used_twice_add():
    check_grads(lambda a: a + a, [A])

def test_same_tensor_used_twice_mul():
    check_grads(lambda a: a * a, [A])

def test_same_tensor_as_base_and_exponent():
    check_grads(lambda a: a ** a, [POS])

def test_diamond_graph_grads():
    # a feeds two branches that rejoin, so its grads have to accumulate from both
    def f(a, b):
        c = a * b
        d = a + b
        return c * d
    check_grads(f, [A, B])

def test_softmax_grads():
    check_grads(lambda a: a.exp() / a.exp().sum(axis=1, keepdims=True) * B, [A])

def test_two_layer_network_grads():
    x = [[1.0, -2.0, 3.0], [0.5, 4.0, -1.5], [2.0, 0.0, 1.0], [-1.0, 1.0, 0.5]]
    w1 = [[0.2, -0.5, 0.1, 0.4], [0.3, 0.8, -0.6, 0.1], [-0.7, 0.2, 0.5, -0.3]]
    b1 = [0.1, -0.2, 0.05, 0.3]
    w2 = [[0.5, -0.4], [0.3, 0.2], [-0.6, 0.1], [0.2, 0.7]]
    b2 = [0.01, -0.02]
    def f(x, w1, b1, w2, b2):
        hidden = (x @ w1 + b1).relu()
        return ((hidden @ w2 + b2).tanh() ** 2).mean()
    check_grads(f, [x, w1, b1, w2, b2])

def test_backward_seeds_output_with_ones():
    out = Tensor(A) * 2.0
    out.backward()
    np.testing.assert_array_equal(out.grads, np.ones((2, 3)))

def test_grads_accumulate_across_backward_calls_on_leaves():
    a, b = Tensor(A), Tensor(B)
    out = a * b
    out.backward()
    first = a.grads.copy()
    out.backward()
    np.testing.assert_allclose(a.grads, 2 * first)

def test_backward_does_not_modify_data():
    a, b = Tensor(A), Tensor(B)
    (a * b + a).relu().backward()
    np.testing.assert_array_equal(a.data, A)
    np.testing.assert_array_equal(b.data, B)


# Known bugs in Tensor: each of these fails today. They are strict xfails, so once the
# implementation is fixed they will start failing as XPASS and the marker should be removed.

@pytest.mark.xfail(strict=True, reason="one non-positive base drops the exponent grads for every element")
def test_pow_exponent_grads_with_mixed_sign_bases():
    a, n = Tensor([2.0, -1.0]), Tensor([2.0, 2.0])
    (a ** n).backward()
    assert n.grads[0] == pytest.approx(4.0 * np.log(2.0))

@pytest.mark.xfail(strict=True, reason="x ** 0 at x == 0 computes 0 * 0 ** -1, which is nan")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_pow_zero_exponent_grads_at_zero_base():
    a = Tensor([0.0, 2.0])
    (a ** 0).backward()
    np.testing.assert_array_equal(a.grads, [0.0, 0.0])

@pytest.mark.xfail(strict=True, reason="1-D operands make .T a no-op, so matmul backward gets the wrong shapes")
def test_matmul_1d_input_grads():
    check_grads(lambda x, w: x @ w, [[1.0, -2.0, 3.0], np.arange(6.0).reshape(3, 2)])

@pytest.mark.xfail(strict=True, reason="no __array_ufunc__ = None, so NumPy handles ndarray * Tensor itself")
def test_ndarray_on_left_returns_tensor():
    assert isinstance(np.array(A) * Tensor(A), Tensor)

@pytest.mark.xfail(strict=True, reason="no __rmatmul__, so ndarray @ Tensor never reaches Tensor")
def test_ndarray_on_left_of_matmul():
    assert isinstance(np.ones((2, 2)) @ Tensor(A), Tensor)

@pytest.mark.xfail(strict=True, reason="np.asarray does not copy, so the Tensor aliases the caller's array")
def test_tensor_does_not_alias_input_array():
    arr = np.array([1.0, 2.0, 3.0])
    t = Tensor(arr)
    arr[0] = 100.0
    assert t.data[0] == 1.0
