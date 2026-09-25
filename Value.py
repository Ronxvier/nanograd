import math
class Value:
    def __init__(self,data,_children=(), _op='',label=''):
        self.data = data;
        self._prev = set(_children) # node keeps track of previous two operators
        self.grad = 0.0
        self._op = _op

    def __repr__(self):
        # For print
        return f"Value(data={self.data})"

    def __add__(self,other):
        # For addition
        out = Value(self.data + other.data, (self,other), '+')
        return out

    def __mul__(self,other):
        # For addition
        out = Value(self.data * other.data, (self,other), '*')
        return out

    def pow(self, other):
        out = Value(self.data ** other.data, (self,other), '**')
        return out
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t,(self, ), 'tanh')
        return out

def intToVal(integer):
    return Value(integer, f'{integer}')

# Testing:
"""
a = Value(2.0,'a')
b = Value(4.0,'b')
c = Value(5.0,'c')
d= a+b*c; d.label = 'd'
print(d._prev) # Here, last two terms are printed (6.0, 5.0)
print(d._op) # +
e = Value(4.0, 'e')
f = a.pow(e)
print(f)
g = f.tanh()
print(g)
"""

