# nanograd

**THIS IS AN AI GENERATED README. It's roughly correct, but I'll go back and write one myself later.**

A small machine learning library built from scratch, mostly to learn how it all works under the hood.

This started as a clone of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd): a scalar-valued autograd engine with a tiny neural network library on top. Now I'm growing it into a more complete ML library.

It's mostly an educational project, but it might still be useful for small experiments, or for reading code that's short enough to follow end to end.

## What's here

| Module | What it does |
| --- | --- |
| `src/engine.py` | `Value`, a scalar that records the operations applied to it and backpropagates gradients through them. Also an early `Tensor` class (work in progress). |
| `src/nn.py` | `Neuron`, `Layer` and `MLP`, built entirely out of `Value`s. |
| `src/draw_graph.py` | Renders a computation graph with Graphviz, showing each node's data and gradient. |

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

draw_dot(loss)   # writes graph.svg and opens it
```

This needs the [Graphviz](https://graphviz.org/download/) system package (for example `brew install graphviz`) as well as the Python `graphviz` package.

## Setup

```bash
git clone https://github.com/<your-username>/nanograd.git
cd nanograd
pip install numpy graphviz pytest
```

## Running the tests

```bash
python -m pytest -s
```

- `tests/test_simplenet.py` trains an MLP on a four-example toy dataset and checks that the loss drops below 0.01.
- `tests/test_tensors.py` exercises the early `Tensor` class.

## Roadmap

Things I'm planning to work through, roughly in order:

- [x] Scalar autograd engine (`Value`)
- [x] Neuron / Layer / MLP
- [x] Computation graph visualization
- [ ] `Tensor` class backed by NumPy, with autograd over whole arrays
- [ ] Full Demo with mnist dataset
- [ ] More activations (ReLU, sigmoid, softmax)
- [ ] Loss functions (MSE, cross-entropy)
- [ ] Optimizers (SGD with momentum, Adam)
- [ ] Training on a real dataset (e.g. MNIST)

## Acknowledgements

- Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and the [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) lectures, which this project is based on. `draw_graph.py` is adapted from the lecture notebook.
