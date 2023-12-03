""" neural network layers """
from catgrad.signature import NdArrayType
from catgrad.combinators import canonical
from catgrad.rdops import *
from catgrad.parameters import parameter

def linear(A: NdArrayType, B: NdArrayType, C: NdArrayType):
    if not (A.dtype == B.dtype and B.dtype == C.dtype):
        raise ValueError("linear: dtypes must be equal, but got dtypes {A.dtype=} {B.dtype=} {C.dtype=}")
    return (identity(obj(A+B)) @ parameter(obj(B+C))) >> op(Compose(A,B,C))

def bias(A: NdArrayType):
    return (parameter(obj(A)) @ identity(obj(A))) >> add(obj(A))

sigmoid = canonical(lambda T: op(Sigmoid(T)))

def dense(A: NdArrayType, B: NdArrayType, C: NdArrayType, activation=sigmoid):
    return linear(A, B, C) >> bias(A+C) >> activation(obj(A+C))
