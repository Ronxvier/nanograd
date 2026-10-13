import numpy as np
import math
class Tensor:
    def __init__(self,data, _children=(), _op='',label=''):
        self.data = np.asarray(data, dtype=np.float64)
        self.grads= np.zeros_like(self.data) # zeros in shape of data
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __add__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        scalar_other = other.data.ndim == 0
        if self.data.shape != other.data.shape and not scalar_other:
            raise ValueError(
                f"Shape mismatch for addition: "
                f"{self.data.shape} and {other.data.shape}"
            )
        out = Tensor(self.data+other.data,(self,other),'+')
        def _backward():
            self.grads += 1.0 * out.grads
            if scalar_other:
                other.grads += np.sum(out.grads)
            else:
                other.grads += 1.0 * out.grads
        out._backward = _backward
        return out

    def __mul__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        scalar_other = (other.data.ndim ==0)
        if not scalar_other and self.data.shape != other.data.shape:
            raise ValueError(
                f"Shape mismatch for multiplication: "
                f"{self.data.shape} and {other.data.shape}"
            )
        out = Tensor(self.data*other.data,(self,other),'*')
        def _backward():
            self.grads += other.data * out.grads
            other_grad = self.data * out.grads
            if scalar_other:
                other.grads += np.sum(other_grad)
            else:
                other.grads += other_grad
        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data,(self,),'neg')
        def _backward():
            self.grads += -out.grads
        out._backward = _backward
        return out

    def __sub__(self, other): # self-other
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        scalar_other = other.data.ndim == 0
        if not scalar_other and self.data.shape != other.data.shape:
            raise ValueError(
                f"Shape mismatch for power: "
                f"{self.data.shape} and {other.data.shape}"
            )

        out = Tensor(self.data ** other.data, (self, other), '**')

        def _backward():
            self.grads += (
                other.data * self.data ** (other.data - 1)
            ) * out.grads

            # restricted to positive bases
            if np.all(self.data > 0):
                exponent_grad = out.data * np.log(self.data) * out.grads
                if scalar_other:
                    other.grads += np.sum(exponent_grad)
                else:
                    other.grads += exponent_grad

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')
        def _backward():
            self.grads += out.data * out.grads
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data,0),(self,),'relu')
        def _backward():
            self.grads += (self.data > 0) * out.grads
        out._backward = _backward
        return out

    def __rmul__(self,other): # other * self
        return self * other
    def __radd__(self,other): # other + self
        return self + other
    def __rsub__(self,other): # other - self
        return other + (-self)
    def __rtruediv__(self,other): # other / self
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grads = np.ones(self.data.shape)
        # topologically sort nodes: left to right
        for node in reversed(topo):
            node._backward()
