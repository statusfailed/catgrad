"""
test some models on a dataset sampled from the following linear regression problem:

    y₀ = 0·x₀ + x₁ + 2x₂ + 3x₃
"""
import ast
import numpy as np

# import catgrad and the python Numpy array backend
from catgrad import *
from catgrad.target.python.array_backend import Numpy


n_examples = 100
min_value = -128
max_value = 128
n_dim = 4

G = np.random.default_rng(1337)
x = G.uniform(low=min_value, high=max_value, size=(n_examples, 4))
true_p = np.arange(n_dim).reshape((n_dim, 1)) # [[0 1 2 3]].T
y = x @ true_p

BATCH_SIZE = n_examples
BATCH_TYPE = NdArrayType((BATCH_SIZE,), Dtype.float32)
INPUT_TYPE = NdArrayType((n_dim,), Dtype.float32)
OUTPUT_TYPE = NdArrayType((1,), Dtype.float32)

def test_linear_model():
    learning_rate = 0.000001
    model = layers.linear(BATCH_TYPE, INPUT_TYPE, OUTPUT_TYPE)

    CompiledModel, ParamType, model_ast = compile_model(model, layers.sgd(learning_rate), layers.mse)
    print(ast.unparse(model_ast))

    compiled_model = CompiledModel(Numpy)
    predict = lambda *args: compiled_model.predict(*args)
    step = lambda *args: compiled_model.step(*args)

    p = [ G.normal(0, 0.01, T.shape).astype(Numpy.dtype(T.dtype)) for T in ParamType ]

    NUM_ITER = 100
    for i in range(0, NUM_ITER):
        # a = model_accuracy(predict, p, x, y)
        # print(i, 100*a, "%")
        p = step(*p, x, y)
        [y_hats] = predict(*p, x)
        print(np.mean(y_hats - y))

    # very small loss
    assert np.mean(y_hats - y) < 1e-10, "loss was unexpectedly high"

    # weights should be 0, 1, 2, 3
    assert np.allclose(p[0], true_p), "weights were not close to [0, 1, 2, 3]"
