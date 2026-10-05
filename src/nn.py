import random
from .engine import Value
class Neuron:
    def __init__(self, nin): # number of inputs
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)] # creates random weights for each input (size defined at initialization, values defined at call)
        self.b = Value(random.uniform(-1,1))
    def __call__(self,x):
        # w * x + b (w * x is a dot product)
        act = sum((wi*xi for wi, xi in zip(self.w,x)),self.b) #(+self.b)
        out = act.tanh()
        return out
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for x in range(nout)] # nout neurons, each with nin weights
    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if (len(outs)==1) else outs
    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params
class MLP:
    def __init__(self, nin, nouts): # nouts defines sizes of all layers
        sz = [nin] + nouts
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]
        """
        # Layer(#of_input, #of_output) (#of_output is = #of_input for next layer)
        """

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()] #TODO: Review this type of list comprehension
