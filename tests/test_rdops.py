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

F = Forget()

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
    arrow = to_python_function(e.arrow())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # the fwd map is an optic with empty residual, so fwd = arrow
    _assert_equal(arrow(x), [x, x])
    _assert_equal(fwd(x), [x, x])
    _assert_equal(rev(x, y), [x + y])

# TODO
@pytest.mark.skip()
def test_rd_ncopy():
    raise NotImplementedError("TODO")

from hypothesis import settings, reproduce_failure
@settings(print_blob=True)
@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=1))
def test_rd_discard(Tx: np.ndarray):
    T, [x] = Tx

    e = Discard(T)
    arrow = to_python_function(e.arrow())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # the fwd map is an optic with empty residual, so fwd = arrow
    _assert_equal(arrow(x), [])
    _assert_equal(fwd(x), [])
    _assert_equal(rev(), [np.zeros(T.shape, Numpy.dtype(T.dtype))])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=3))
def test_rd_add(Tx):
    T, [x0, x1, y] = Tx

    e = Add(T)

    arrow = to_python_function(e.arrow())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # the fwd map is an optic with empty residual, so fwd = arrow
    _assert_equal(arrow(x0, x1), [x0 + x1])
    _assert_equal(fwd(x0, x1), [x0 + x1])
    _assert_equal(rev(y), [y, y])

@pytest.mark.skip()
def test_rd_nadd():
    raise NotImplementedError("TODO")

@given(ndarrays(array_type=ndarraytypes(shape=st.just(()))))
def test_rd_constant(Tx: np.ndarray):
    T, [x] = Tx

    e = Constant(T, x.item())

    arrow = to_python_function(e.arrow())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # the fwd map is an optic with empty residual, so fwd = arrow
    _assert_equal(arrow(), [x])
    _assert_equal(fwd(), [x])
    _assert_equal(rev(x), [])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=3))
def test_rd_multiply(Tx: np.ndarray):
    T, [x0, x1, y] = Tx

    e = Multiply(T)

    arrow = to_python_function(e.arrow())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # the fwd map is an optic with empty residual, so fwd = arrow
    _assert_equal(arrow(x0, x1), [x0 * x1])
    _assert_equal(fwd(x0, x1), [x0 * x1, x0, x1]) # this one's a lens
    _assert_equal(rev(x0, x1, y), [x1 * y, x0 * y])


################################################################################
# TODO
################################################################################

@pytest.mark.skip()
@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(composable_ndarrays())
def test_rd_compose(ABCxy: np.ndarray):
    A, B, C, x, y = ABCxy
    e = Compose(A, B, C)

    arrow = to_python_function(e.arrow())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    expected_y = np.tensordot(x0, x1, axes=len(B.shape))
    expected_dx0 = np.tensordot(y, x1, axes=len(C.shape))
    expected_dx1 = np.tensordot(x0, y, axes=len(A.shape))

    # the fwd map is an optic with empty residual, so fwd = arrow
    _assert_equal(arrow(x0, x1), [expected_y])
    _assert_equal(fwd(x0, x1), [expected_y, x0, x1]) # this one's a lens
    _assert_equal(rev(x0, x1, y), [expected_dx0, expected_dx1])


@pytest.mark.skip()
@given(reshape_args())
def test_reshape(XYx):
    X, Y, x = XYx
    f = to_python_function(op(Reshape(X, Y)))
    _assert_equal(f(x), [x.reshape(Y.shape)])

@pytest.mark.skip()
@given(ndarrays())
def test_transpose(Tx):
    T, [x] = Tx
    f = to_python_function(op(Transpose(T)))
    _assert_equal(f(x), [x.transpose()])
