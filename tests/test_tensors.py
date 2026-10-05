from src.engine import *

def test_TensorOps():
    t = Tensor([[[1,2],[1,2]],[[1,2],[1,2]],[[1,2],[1,2]],[[1,2],[1,2]]])
    print(t.shape())
