""" Test that the Python backend behaves as expected """
import pytest
import numpy as np
from typing import List, Tuple
from hypothesis import given
from hypothesis import strategies as st
from tests.strategies import ndarrays, ndarraytypes, composable_ndarrays, reshape_args
import tests.strategies as strategies

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

@pytest.mark.skip()
def test_rd_ncopy():
    # TODO!
    raise NotImplementedError("TODO")

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

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(composable_ndarrays())
def test_rd_compose(ABCxy: np.ndarray):
    A, B, C, x0, x1 = ABCxy
    # TODO: generate a random dy
    dy = np.ones((A+C).shape, Numpy.dtype(A.dtype))
    e = Compose(A, B, C)

    arrow = to_python_function(e.arrow())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # Work out axes to swap
    # TODO: why are the arguments backwards here? (& in Compose.rev())
    t_AB = FiniteFunction.twist(len(B.shape), len(A.shape)).table.tolist()
    t_BC = FiniteFunction.twist(len(C.shape), len(B.shape)).table.tolist()
    x0_dagger = np.transpose(x0, t_AB)
    x1_dagger = np.transpose(x1, t_BC)

    expected_y = np.tensordot(x0, x1, axes=len(B.shape))
    expected_dx0 = np.tensordot(dy, x1_dagger, axes=len(C.shape))
    expected_dx1 = np.tensordot(x0_dagger, dy, axes=len(A.shape))

    # the fwd map is an optic with empty residual, so fwd = arrow
    _assert_equal(arrow(x0, x1), [expected_y])
    _assert_equal(fwd(x0, x1), [expected_y, x0, x1]) # this one's a lens
    _assert_equal(rev(x0, x1, dy), [expected_dx0, expected_dx1])

@given(reshape_args())
def test_rd_reshape(XYx):
    X, Y, x = XYx
    e = Reshape(X, Y)

    arrow = to_python_function(e.arrow())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    expected_y = x.reshape(Y.shape)

    _assert_equal(arrow(x), [expected_y])
    _assert_equal(fwd(x), [expected_y]) # this one's a lens
    _assert_equal(rev(expected_y), [x]) # inverse

@given(strategies.permute().flatmap(lambda op: st.tuples(st.just(op), ndarrays(array_type=st.just(op.T)))))
def test_rd_permute(p_x: Tuple[ops.Permute, np.ndarray]):
    p, (_, [x]) = p_x
    e = Permute(p.T, p.p) # "upcast" to rdop

    arrow = to_python_function(e.arrow())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    expected_y = x.transpose(e.p)

    _assert_equal(arrow(x), [expected_y])
    _assert_equal(fwd(x), [expected_y]) # this one's a lens
    _assert_equal(rev(expected_y), [x]) # inverse
