import numpy as np
import pytest
from hypothesis import given, example

from tests.utils import assert_equal
import tests.strategies as strategies

from open_hypergraphs import FiniteFunction, OpenHypergraph

from catgrad.target.python import to_python_function
from catgrad.target.python.array_backend import Numpy
from catgrad.signature import obj, NdArrayType, Dtype
from catgrad.rdops import Forget, copy, discard, add, zero, multiply, constant

from catgrad.functor import Bidirectional

F = Forget()
B = Bidirectional()

from hypothesis import reproduce_failure
@example(obj(NdArrayType((), Dtype.int32), NdArrayType((), Dtype.float32)))
@given(strategies.objects())
def test_bidirectional_zero(A: FiniteFunction):
    c = zero(A)
    Bc = B(c)

    assert Bc.source == FiniteFunction.transpose(1, len(c.source)) >> (c.source + c.source)
    assert Bc.target == FiniteFunction.transpose(2, len(c.target)) >> (c.target + c.target)

    d = F(B.adapt(Bc, c.source, c.target))
    f = to_python_function(d) # TODO: REMOVE THIS GARBAGE HACK

    # TODO: get random args from hypothesis
    args = [ np.ones(shape=T.shape, dtype=Numpy.dtype(T.dtype)) for T in A ]
    expected = [ np.zeros(shape=T.shape, dtype=Numpy.dtype(T.dtype)) for T in c.target ]
    actual = f(*args)
    assert_equal(expected, actual)
