import math
class Value:
    def __init__(self,data,_children=(), _op='',label=''):
        self.data = data;
        self._prev = set(_children) # node keeps track of previous two operators
        self._backward = lambda: None # Default is empty function
        self.grad = 0.0
        self._op = _op

    def __repr__(self):
        # For print
        return f"Value(data={self.data})"

    def __truediv__(self, other):
        return self * other ** -1
    def __neg__(self):
        return self * -1
    def __sub__(self, other): # self-other
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __add__(self,other):
        # For addition
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self,other), '+')
        def _backward():
            # In the case of addition, the addends take on the gradient of their resulting value, as they don't change the gradient of the resulting value.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out



    def __mul__(self,other):
        # For addition
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self,other), '*')
        def _backward():
            self.grad += other.data * out.grad # chain rule
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self,other), f'**{other}')
        def _backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            # TODO: should probably compute other grad too
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(self.data), (self, ), 'exp')
        def _backward():
            self.grad += out.data * out.grad # e^a a'
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t,(self, ), 'tanh')
        def _backward():
            self.grad += (1-t**2)*out.grad
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
        self.grad = 1.0
        # topologically sort nodes: left to right
        for node in reversed(topo):
            node._backward()
