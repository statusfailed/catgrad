import ast
import argparse
import numpy as np
import pandas as pd
import torch

# import catgrad and the python Numpy array backend
from catgrad import *
from catgrad.target.python.array_backend import Numpy
from catgrad.target.python.array_backend import Torch

# Tensor types: Iris has a 4-dimensional input and is a 3-class classification
# problem.
INPUT_TYPE = NdArrayType((4,), Dtype.float32)
OUTPUT_TYPE = NdArrayType((3,), Dtype.float32)

# compute accuracy of predictions given true labels
def accuracy(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    num = np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)
    den = len(y_true)
    if isinstance(num, torch.Tensor):
        num = num.numpy()
    return np.sum(num) / den

# Compute accuracy of the trained model predict(p, -).
def model_accuracy(predict, p, x, y):
    [y_hats] = predict(*p, x)
    return accuracy(y_hats, y)

def load_iris(path, use_torch):
    iris = pd.read_csv(path)

    # load training data
    train_input = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()

    # construct labels manually since data is ordered by class
    train_labels = np.array([0]*50 + [1]*50 + [2]*50).reshape(-1)

    # one-hot encode 3 classes
    train_labels = np.identity(3)[train_labels]

    if use_torch:
        train_input = torch.tensor(train_input, dtype=torch.float32)
        train_labels = torch.tensor([0]*50 + [1]*50 + [2]*50).reshape(-1)
        train_labels = torch.eye(3)[train_labels]

    return train_input, train_labels

def main():
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--iris-data', default='data/iris.csv')
    parser.add_argument('--use-torch', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('model', default='linear')
    args = parser.parse_args()

    BATCH_SIZE = 150
    BATCH_TYPE = NdArrayType((BATCH_SIZE,), Dtype.float32)
    match args.model:
        case 'linear':
            # big batch sizes mean we have to set a lower learning rate for linear model;
            # otherwise it 'oscillates' around the one-hot-encoded target.
            learning_rate = 0.0001
            model = layers.linear(BATCH_TYPE, INPUT_TYPE, OUTPUT_TYPE)
        case 'simple':
            learning_rate = 0.01
            model = layers.linear(BATCH_TYPE, INPUT_TYPE, OUTPUT_TYPE) >> layers.sigmoid(obj(BATCH_TYPE + OUTPUT_TYPE))
        case 'dense':
            learning_rate = 0.01
            model = layers.dense(BATCH_TYPE, INPUT_TYPE, OUTPUT_TYPE, activation=layers.sigmoid)
        case 'hidden':
            learning_rate = 0.01
            # hidden layer size
            HIDDEN_TYPE = NdArrayType((20,), Dtype.float32)
            model = layers.dense(BATCH_TYPE, INPUT_TYPE, HIDDEN_TYPE, activation=layers.sigmoid) \
                    >> layers.dense(BATCH_TYPE, HIDDEN_TYPE, OUTPUT_TYPE, activation=layers.sigmoid)
        case _:
            raise ValueError(f"unknown model type: {args.model}")

    # Compile to Python + print the source
    CompiledModel, ParamType, model_ast = compile_model(model, layers.sgd(learning_rate), layers.mse)
    print(ast.unparse(model_ast))
    # Instantiate compiled model with a backend (Numpy)
    if args.use_torch:
        compiled_model = CompiledModel(Torch)
    else:
        compiled_model = CompiledModel(Numpy)

    # wrap predict and step for convenience.
    predict = lambda *args: compiled_model.predict(*args)
    step = lambda *args: compiled_model.step(*args)

    # do param initialization manually for now.
    if args.use_torch:
        p = [torch.randn(size=T.shape, dtype=Torch.dtype(T.dtype))*0.01 for T in ParamType]
    else:
        p = [np.random.normal(0, 0.01, T.shape).astype(Numpy.dtype(T.dtype)) for T in ParamType]

    # Load data from CSV
    print("loading data...")
    train_input, train_labels = load_iris(args.iris_data, args.use_torch)

    print("training...")
    N = len(train_input)

    x = train_input
    y = train_labels

    # train loop with compiled "predict" and "step" functions.
    # we print accuracy every iteration, but that's obviously not a great idea!
    NUM_ITER = 1000
    for i in range(0, NUM_ITER):
        # print accuracy at each iteration
        a = model_accuracy(predict, p, x, y)
        print(i, 100*a, "%")

        # update params
        p = step(*p, x, y)

    print("predicting...")
    [y_hats] = predict(*p, x)
    print(f'accuracy: {100*accuracy(y_hats, y)}%')

if __name__ == "__main__":
    main()
