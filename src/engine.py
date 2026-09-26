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

    def __add__(self,other):
        # For addition
        out = Value(self.data + other.data, (self,other), '+')
        def _backward():
            # In the case of addition, the addends take on the gradient of their resulting value, as they don't change the gradient of the resulting value.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self,other):
        # For addition
        out = Value(self.data * other.data, (self,other), '*')
        def _backward():
            self.grad += other.data * out.grad # chain rule
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def pow(self, other):
        out = Value(self.data ** other.data, (self,other), '**')
        def _backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            other.grad += (self.data ** other.data) * math.log(self.data) * out.grad
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
        # topologically sort nodes: left to right
        for node in reversed(topo):
            node._backward()


def intToVal(integer):
    custom = Value(integer)
    custom.label = f"{integer}"
    return custom

