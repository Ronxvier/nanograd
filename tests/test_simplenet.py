from src.nn import MLP

def test_mlp_learns_toy_dataset():
    # Build mlp
    n = MLP(3,[4, 4, 1])

    """
    input: 3 numbers
      -> Layer 1: 4 neurons × 3 weights each  -> 4 values
      -> Layer 2: 4 neurons × 4 weights each  -> 4 values
      -> Layer 3: 1 neuron  × 4 weights       -> 1 value
    """

    xs = [
        [2.0, 3.0, -1.0], #1
        [3.0, -1.0, 0.5], #0
        [0.5, 1.0, 1.0], #0
        [1.0, 1.0, 1.0], #1
    ]

    ys = [1.0, -1.0, -1.0, 1.0] # desired outputs

    for k in range(1001):
        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum((yout-ygt)**2 for ygt, yout in zip(ys, ypred))# mean squared error loss

        # backward pass
        for p in n.parameters():
            p.grad = 0 #zerograd
        loss.backward()

        # update
        for p in n.parameters():
            p.data += -0.05 * p.grad # (we want to decrease the gradient, move in negative loss' direction) Too big of a step can cause an overcorrection.
        if k % 100 == 0: print(k, loss.data)
    print(ypred)

    # predictions should be close to the targets
    assert loss.data < 0.01
