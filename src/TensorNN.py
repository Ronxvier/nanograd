import random
import numpy as np
from .Tensor import Tensor

class Layer:
    def __init__(self, nin, nout):
        weights = [random.uniform(-1,1) for _ in range(nin)] # todo: validate
        self.w = Tensor(weights )
        self.b = Tensor(np.zeros(nout))
    def __call__(self,x):
        act = self.w @ x + self.b # Todo: validate
        outs = act.tanh()
        return outs
    def parameters(self):
        return [self.w , self.b]

