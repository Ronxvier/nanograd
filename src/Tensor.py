import numpy as np
import math
class Tensor:
    def __init__(self,data, _children=(), _op='',label=''):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad= np.zeros_like(self.data) # zeros in shape of data
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __add__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.data.shape != other.data.shape:
            raise ValueError(
                f"Shape mismatch for addition: "
                f"{self.data.shape} and {other.data.shape}"
            )
        out = Tensor(self.data+other.data,(self,other),'+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.data.shape != other.data.shape:
            raise ValueError(
                f"Shape mismatch for multiplication: "
                f"{self.data.shape} and {other.data.shape}"
            )
        #TODO: need to implement a method for scalar multiplication (t * 2 type operations), This can just be an exception
        out = Tensor(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data,(self,),'neg')
        def _backward():
            self.out += -other.grad
        out._backward = _backward
        return out

    def __sub__(self, other): # self-other
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)
