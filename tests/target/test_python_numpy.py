""" Test that the Python backend behaves as expected """
import pytest
import numpy as np
from typing import List
from hypothesis import given
from hypothesis import strategies as st
from tests.strategies import ndarrays, ndarraytypes, composable_ndarrays, reshape_args

from catgrad.signature import NdArrayType
from catgrad.target.python import to_python_function
from catgrad.operations import *

def _assert_equal(xs: List[np.ndarray], ys: List[np.ndarray]):
    assert len(xs) == len(ys)
    for x, y in zip(xs, ys):
        assert np.array_equal(x, y, equal_nan=True)

@given(ndarrays())
def test_copy(Tx: np.ndarray):
    T, [x] = Tx
    f = to_python_function(Copy(T).op)
    assert f(x) == [x, x]

@given(ndarrays())
def test_discard(Tx: np.ndarray):
    T, [x] = Tx
    f = to_python_function(Discard(T).op)
    assert f(x) == []

# we're just testing against reference impl; overflow/nan is fine.
@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=2))
def test_add(Tx: np.ndarray):
    T, [x, y] = Tx
    f = to_python_function(Add(T).op)
    _assert_equal(f(x, y), [x+y])

@given(ndarrays(array_type=ndarraytypes(shape=st.just(()))))
def test_constant(Tx: np.ndarray):
    T, [x] = Tx
    f = to_python_function(Constant(T, x.item()).op)
    _assert_equal(f(), [x])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=2))
def test_multiply(Tx: np.ndarray):
    T, [x, y] = Tx
    f = to_python_function(Multiply(T).op)
    _assert_equal(f(x, y), [x * y])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(composable_ndarrays())
def test_compose(ABCxy: np.ndarray):
    A, B, C, x, y = ABCxy
    f = to_python_function(Compose(A, B, C).op)
    _assert_equal(f(x, y), [np.tensordot(x, y, axes=len(B.shape))])

@given(reshape_args())
def test_reshape(XYx):
    X, Y, x = XYx
    f = to_python_function(Reshape(X, Y).op)
    _assert_equal(f(x), [x.reshape(Y.shape)])

@given(ndarrays())
def test_transpose(Tx):
    T, [x] = Tx
    f = to_python_function(Transpose(T).op)
    _assert_equal(f(x), [x.transpose()])
