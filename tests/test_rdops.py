""" Test that the Python backend behaves as expected """
import pytest
import numpy as np
from typing import List, Tuple
from hypothesis import given
from hypothesis import strategies as st

from tests.utils import assert_equal
from tests.strategies import ndarrays, ndarraytypes, composable_ndarrays, reshape_args, ncopy_args
import tests.strategies as strategies

from catgrad.signature import NdArrayType
from catgrad.target.python import to_python_function
from catgrad.target.python.array_backend import Numpy
from catgrad.rdops import *

F = Forget()

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
    assert_equal(arrow(x), [x, x])
    assert_equal(fwd(x), [x, x])
    assert_equal(rev(x, y), [x + y])

@given(ncopy_args())
def test_rd_ncopy(TxN):
    (T, [x]), N = TxN
    e = NCopy(N, T)

    arrow = to_python_function(e.arrow())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    dy = np.ones((N+T).shape) # TODO: generate
    expected_y = np.broadcast_to(x, (N+T).shape)

    assert_equal(arrow(x), [expected_y])
    assert_equal(fwd(x), [expected_y])
    assert_equal(rev(dy), [Numpy.nadd(N.shape, dy)])

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
    assert_equal(arrow(x), [])
    assert_equal(fwd(x), [])
    assert_equal(rev(), [np.zeros(T.shape, Numpy.dtype(T.dtype))])

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
    assert_equal(arrow(x0, x1), [x0 + x1])
    assert_equal(fwd(x0, x1), [x0 + x1])
    assert_equal(rev(y), [y, y])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays())
def test_rd_nadd(Tx: np.ndarray):
    # TODO: test more than a single extra dimension!
    T, xs = Tx
    x = np.stack(xs)
    N = NdArrayType((len(xs),), T.dtype)
    e = NAdd(N, T)

    arrow = to_python_function(e.arrow())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    dy = np.ones((T).shape) # TODO: generate
    expected_y = Numpy.nadd(N.shape, x)

    assert_equal(arrow(x), [expected_y])
    assert_equal(fwd(x), [expected_y])
    assert_equal(rev(dy), [Numpy.ncopy(N.shape, dy)])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=3))
def test_rd_subtract(Tx):
    T, [x0, x1, y] = Tx

    e = Subtract(T)

    arrow = to_python_function(e.arrow())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # the fwd map is an optic with empty residual, so fwd = arrow
    assert_equal(arrow(x0, x1), [x0 - x1])
    assert_equal(fwd(x0, x1), [x0 - x1])
    assert_equal(rev(y), [y, -y])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=2))
def test_rd_negate(Tx):
    T, [x, dy] = Tx

    e = Negate(T)

    arrow = to_python_function(e.arrow())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # the fwd map is an optic with empty residual, so fwd = arrow
    assert_equal(arrow(x), [-x])
    assert_equal(fwd(x), [-x])
    assert_equal(rev(dy), [-dy])

@given(ndarrays(array_type=ndarraytypes(shape=st.just(()))))
def test_rd_constant(Tx: np.ndarray):
    T, [x] = Tx

    e = Constant(T, x.item())

    arrow = to_python_function(e.arrow())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # the fwd map is an optic with empty residual, so fwd = arrow
    assert_equal(arrow(), [x])
    assert_equal(fwd(), [x])
    assert_equal(rev(x), [])

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
    assert_equal(arrow(x0, x1), [x0 * x1])
    assert_equal(fwd(x0, x1), [x0 * x1, x0, x1]) # this one's a lens
    assert_equal(rev(x0, x1, y), [x1 * y, x0 * y])

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
    assert_equal(arrow(x0, x1), [expected_y])
    assert_equal(fwd(x0, x1), [expected_y, x0, x1]) # this one's a lens
    assert_equal(rev(x0, x1, dy), [expected_dx0, expected_dx1])

@given(reshape_args())
def test_rd_reshape(XYx):
    X, Y, x = XYx
    e = Reshape(X, Y)

    arrow = to_python_function(e.arrow())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    expected_y = x.reshape(Y.shape)

    assert_equal(arrow(x), [expected_y])
    assert_equal(fwd(x), [expected_y]) # this one's a lens
    assert_equal(rev(expected_y), [x]) # inverse

@given(strategies.permute().flatmap(lambda op: st.tuples(st.just(op), ndarrays(array_type=st.just(op.T)))))
def test_rd_permute(p_x: Tuple[ops.Permute, np.ndarray]):
    p, (_, [x]) = p_x
    e = Permute(p.T, p.p) # "upcast" to rdop

    arrow = to_python_function(e.arrow())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    expected_y = x.transpose(e.p)

    assert_equal(arrow(x), [expected_y])
    assert_equal(fwd(x), [expected_y]) # this one's a lens
    assert_equal(rev(expected_y), [x]) # inverse
