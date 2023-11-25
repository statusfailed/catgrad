""" Test that the Python backend behaves as expected """
import pytest
import numpy as np
from typing import List, Tuple
from hypothesis import given
from hypothesis import strategies as st
from tests.strategies import dtypes, shapes, ndarrays, ndarraytypes, composable_ndarrays, reshape_args, ncopy_args
import tests.strategies as strategies

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
    f = to_python_function(op(Copy(T)))
    _assert_equal(f(x), [x, x])

@given(ncopy_args())
def test_ncopy(TxN):
    (T, [x]), N = TxN
    f = to_python_function(op(NCopy(N, T)))

    actual = f(x)
    expected = [np.broadcast_to(x, (N+T).shape)]
    _assert_equal(actual, expected)

@given(ndarrays())
def test_discard(Tx: np.ndarray):
    T, [x] = Tx
    f = to_python_function(op(Discard(T)))
    _assert_equal(f(x), [])

# we're just testing against reference impl; overflow/nan is fine.
@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=2))
def test_add(Tx: np.ndarray):
    T, [x, y] = Tx
    f = to_python_function(op(Add(T)))
    _assert_equal(f(x, y), [x+y])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays())
def test_nadd(Tx: np.ndarray):
    # TODO: test more than a single extra dimension!
    # instead of n: int, have ndarrays include n: shape, e.g. (2,3,4), then generate 2*3*4 ndarrays.
    T, xs = Tx
    N = NdArrayType((len(xs),), T.dtype)
    f = to_python_function(op(NAdd(N, T)))

    x = np.stack(xs)
    actual = f(x)
    expected = [x.sum(axis=0)] # TODO: more dims
    _assert_equal(actual, expected)

@given(ndarrays(array_type=ndarraytypes(shape=st.just(()))))
def test_constant(Tx: np.ndarray):
    T, [x] = Tx
    f = to_python_function(op(Constant(T, x.item())))
    _assert_equal(f(), [x])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=2))
def test_multiply(Tx: np.ndarray):
    T, [x, y] = Tx
    f = to_python_function(op(Multiply(T)))
    _assert_equal(f(x, y), [x * y])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(composable_ndarrays())
def test_compose(ABCxy: np.ndarray):
    A, B, C, x, y = ABCxy
    f = to_python_function(op(Compose(A, B, C)))
    _assert_equal(f(x, y), [np.tensordot(x, y, axes=len(B.shape))])

@given(reshape_args())
def test_reshape(XYx):
    X, Y, x = XYx
    f = to_python_function(op(Reshape(X, Y)))
    _assert_equal(f(x), [x.reshape(Y.shape)])

@given(ndarrays())
def test_transpose(Tx):
    T, [x] = Tx
    p = list(reversed(range(len(T.shape))))
    f = to_python_function(op(Permute(T, p)))
    _assert_equal(f(x), [x.transpose()])

@given(strategies.permute().flatmap(lambda op: st.tuples(st.just(op), ndarrays(array_type=st.just(op.T)))))
def test_permute(p_x: Tuple[Permute, np.ndarray]):
    p, (_, [x]) = p_x
    f = to_python_function(op(p))

    actual = f(x)
    expected = [x.transpose(p.p)]
    _assert_equal(actual, expected)
