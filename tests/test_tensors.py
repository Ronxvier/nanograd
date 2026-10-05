from src.engine import *
import numpy as np

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
