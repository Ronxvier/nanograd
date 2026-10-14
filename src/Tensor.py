import numpy as np
import math

def _unbroadcast(grad, shape):
  while grad.ndim > len(shape):
      grad = grad.sum(axis=0)
  for i, n in enumerate(shape):
      if n == 1 and grad.shape[i] != 1:
          grad = grad.sum(axis=i, keepdims=True)
  return grad

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
        out = Tensor(self.data+other.data,(self,other),'+')
        def _backward():
            self.grads += 1.0 * _unbroadcast(out.grads, self.grads.shape)
            other.grads += 1.0 * _unbroadcast(out.grads, other.grads.shape)
        out._backward = _backward
        return out

    def __mul__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data*other.data,(self,other),'*')
        def _backward():
            self.grads += _unbroadcast(other.data * out.grads, self.grads.shape)
            other_grad = self.data * out.grads
            other.grads += _unbroadcast(self.data*out.grads, other.grads.shape)
        out._backward = _backward
        return out

    def __matmul__(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data@other.data, (self,other),'@')

        def _backward():
            self.grads += out.grads @ other.data.T # other tensor transpose
            other.grads += self.data.T @ out.grads #self tensor transpose (Got this off definition: TODO: look into this proof)
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
        out = Tensor(self.data ** other.data, (self, other), '**')
        def _backward():
            self.grads += _unbroadcast(
                (other.data * self.data ** (other.data - 1)
            * out.grads), self.grads.shape)
            # restricted to positive bases
            if np.all(self.data > 0):
                exponent_grad = out.data * np.log(self.data) * out.grads
                other.grads += _unbroadcast(exponent_grad,other.grads.shape)
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
