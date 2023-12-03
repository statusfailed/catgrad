# catgrad

You like category theory? You like tinygrad? You love catgrad! ❤️

catgrad is a bit different: instead of using autograd to train, you *compile*
your model's reverse pass into static code.
This means your training loop can run without needing a deep learning framework
(not even catgrad!)

Here is a linear model in `catgrad`:

    model = layers.linear(BATCH_TYPE, INPUT_TYPE, OUTPUT_TYPE)

catgrad can compile this model into static python code:

    class CompiledModel:
        backend: ArrayBackend

        def predict(self, x1, x0):
            x2 = x0 @ x1
            return [x2]

        def step(self, x0, x1, x9):
            x4, x10 = (x0, x0)
            x11, x12 = (x1, x1)
            x16 = self.backend.constant(0.0001, Dtype.float32)
            # ... snip ...
            x18 = x17 * x5
            x2 = x10 - x18
            return [x2]

... so you can train your model by just iterating `step`; no autograd needed:

    for i in range(0, NUM_ITER):
        p = step(p, x, y)

Catgrad uses [reverse derivatives](https://arxiv.org/abs/1910.07065)
and [open hypergraphs](https://github.com/statusfailed/open-hypergraphs/)
to transform a model into its backwards pass.
For details, see [this paper](https://arxiv.org/abs/2305.01041).

# Install

    pip install catgrad

# Examples

Train simple MLPs for the
[iris dataset](https://archive.ics.uci.edu/dataset/53/iris):

    ./data/get-iris-data.sh
    python3 -m examples.iris (linear|simple|dense|hidden)

# Compilation Targets

Target backends we plan to support soon:

- [x] Python/numpy
- [ ] Python/[tinygrad](https://github.com/tinygrad/tinygrad/)
- [ ] C++/[GGML](https://github.com/ggerganov/ggml)
