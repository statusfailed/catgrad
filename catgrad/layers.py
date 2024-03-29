""" neural network layers """
from catgrad.signature import NdArrayType
from catgrad.combinators import canonical
from catgrad.bidirectional.operation import *
from catgrad.special.parameter import parameter

def flatten(X: NdArrayType, Y: NdArrayType):
    return op(Reshape(X, Y))

# linear is OK when you have a *batch* of A inputs of type (1 → B) and you want to multiply them all by the same matrix
# (B ⇒ C) to get a batch of A outputs (A ⇒ C).
# If you need to "pointwise batch", use "batch_linear"
def linear(A: NdArrayType, B: NdArrayType, C: NdArrayType):
    if not (A.dtype == B.dtype and B.dtype == C.dtype):
        raise ValueError(f"linear: dtypes must be equal, but got dtypes {A.dtype=} {B.dtype=} {C.dtype=}")
    return (identity(obj(A+B)) @ parameter(obj(B+C))) >> op(Compose(A,B,C))

def bias(A: NdArrayType):
    return (parameter(obj(A)) @ identity(obj(A))) >> add(obj(A))

sigmoid = canonical(lambda T: op(Sigmoid(T)))

def dense(A: NdArrayType, B: NdArrayType, C: NdArrayType, activation=sigmoid):
    return linear(A, B, C) >> bias(A+C) >> activation(obj(A+C))
