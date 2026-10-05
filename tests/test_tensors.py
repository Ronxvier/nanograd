from src.engine import *

def test_TensorOps():
    t = Tensor([[[1,2],[1,2]],[[1,2],[1,2]],[[1,2],[1,2]],[[1,2],[1,2]]])
    print(t.shape())
    assert t.shape() == (4, 2, 2), "Incorrect Shape"
    t.toValue()
    checkWrap(t.values)
    print(t)

def checkWrap(x):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return [checkWrap(el) for el in x]
    elif isinstance(x, Value):
        return x
    else:
        raise ValueError("Non-Value element found in wrapped Tensor.")
