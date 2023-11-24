""" Test that the Python backend behaves as expected """
import pytest
import numpy as np
from typing import List
from hypothesis import given
from hypothesis import strategies as st
from tests.strategies import ndarrays, ndarraytypes, composable_ndarrays, reshape_args

from catgrad.signature import NdArrayType
from catgrad.target.python import to_python_function
from catgrad.target.python.array_backend import Numpy
from catgrad.rdops import *

def _assert_equal(xs: List[np.ndarray], ys: List[np.ndarray]):
    assert len(xs) == len(ys)
    for x, y in zip(xs, ys):
        assert np.array_equal(x, y, equal_nan=True)

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=2))
def test_rd_copy(Tx: np.ndarray):
    T, [x, y] = Tx

    e = Copy(T)
    F = Forget()
    arrow = to_python_function(e.arrow())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # the fwd map is an optic with empty residual, so fwd = arrow
    _assert_equal(arrow(x), [x, x])
    _assert_equal(fwd(x), [x, x])
    _assert_equal(rev(x, y), [x + y])

from hypothesis import settings, reproduce_failure
@settings(print_blob=True)
@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=2))
def test_rd_discard(Tx: np.ndarray):
    T, [x, y] = Tx

    e = Discard(T)
    F = Forget()
    arrow = to_python_function(e.arrow())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # the fwd map is an optic with empty residual, so fwd = arrow
    _assert_equal(arrow(x), [])
    _assert_equal(fwd(x), [])
    _assert_equal(rev(), [np.zeros(T.shape, Numpy.dtype(T.dtype))])
