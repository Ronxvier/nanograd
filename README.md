# nanograd

A "deep learning library from scratch" pet project.

### Autograd engine

`Value` supports `+`, `-`, `*`, `/`, `**`, `exp()` and `tanh()`, including mixed expressions with plain Python numbers (`2 * x`, `1 - y`, and so on). Calling `.backward()` on an output topologically sorts the graph and applies the chain rule in reverse, storing each node's gradient in `.grad`.

```python
from src.engine import Value

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

d = a * b + c        # 4.0
e = d.tanh()
e.backward()

print(a.grad, b.grad, c.grad)
```

### Neural networks

An `MLP` takes the number of inputs and a list of layer sizes. Every neuron uses a `tanh` activation.

```python
from src.nn import MLP

model = MLP(3, [4, 4, 1])   # 3 inputs -> 4 -> 4 -> 1 output

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, 1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

for step in range(1000):
    # forward pass + mean squared error
    preds = [model(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))

    # backward pass
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    # gradient descent
    for p in model.parameters():
        p.data -= 0.05 * p.grad
```

### Visualizing the graph

```python
from src.draw_graph import draw_dot

draw_dot(loss)
```

This needs the [Graphviz](https://graphviz.org/download/) system package (`brew install graphviz`) and Python `graphviz` package.

## TODO

- [x] Scalar autograd engine (`Value`)
- [x] Neuron / Layer / MLP
- [x] Computation graph visualization
- [ ] Implement Tensors w/ autograd - **IN PROGRESS**
- [ ] Layers, Losses
- [ ] Training on a real dataset (e.g. MNIST)
- [ ] Optimizers (SGD with momentum, Adam)

## Acknowledgements

- Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and the [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) lectures, which this project is based on. `draw_graph.py` is adapted from the lecture notebook.
