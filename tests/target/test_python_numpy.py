""" Test that the Python backend behaves as expected """
import pytest
import numpy as np
from typing import List, Tuple
from hypothesis import given
from hypothesis import strategies as st

from tests.utils import assert_equal
from tests.strategies import dtypes, shapes, ndarrays, ndarraytypes, composable_ndarrays, reshape_args, ncopy_args, nadd_args, nmax_args, nsplit_args, matrix_multiply_args
import tests.strategies as strategies

from catgrad.signature import NdArrayType
from catgrad.target.python import to_python_function
from catgrad.core.operation import *

@given(ndarrays())
def test_copy(Tx: np.ndarray):
    T, [x] = Tx
    f = to_python_function(op(Copy(T)))
    assert_equal(f(x), [x, x])

@given(ncopy_args())
def test_ncopy(NxT):
    (N, [x]), T = NxT
    f = to_python_function(op(NCopy(N, T)))

    actual = f(x)
    expected = [np.broadcast_to(x.reshape(N.shape + (1,)*len(T.shape)), (N+T).shape)]
    assert_equal(actual, expected)

@given(ndarrays())
def test_discard(Tx: np.ndarray):
    T, [x] = Tx
    f = to_python_function(op(Discard(T)))
    assert_equal(f(x), [])

@given(nsplit_args())
def test_nsplit(Tkx):
    (T, k, x) = Tkx
    assert x.shape == (T.shape + (k,))
    c = op(NSplit(T, k))
    f = to_python_function(c)

    actual = f(x)
    expected = np.split(x, k, -1) # split the last axis into k
    assert_equal(actual, expected)

@given(nsplit_args())
def test_nconcatenate(Tkx):
    # nconcatenate args are just nsplit args... split
    (T, k, x) = Tkx
    assert x.shape == (T.shape + (k,))

    # a list of args; split along last axis
    xs = np.split(x, k, -1)

    c = op(NConcatenate(T, k))
    f = to_python_function(c)

    actual = f(*xs)
    expected = [x] # the original concatenated array
    assert_equal(actual, expected)


# we're just testing against reference impl; overflow/nan is fine.
@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=2))
def test_add(Tx: np.ndarray):
    T, [x, y] = Tx
    f = to_python_function(op(Add(T)))
    assert_equal(f(x, y), [x+y])

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
    dims = tuple( -(i+1) for i in reversed(range(len(T.shape))) )
    expected = [x.sum(dims)]
    assert_equal(actual, expected)


@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(nmax_args())
def test_nmax(NTx: np.ndarray):
    N, T, x = NTx

    f = to_python_function(op(NMax(N, T)))

    actual = f(x)
    dims = tuple( -(i+1) for i in reversed(range(len(T.shape))) )
    expected = [x.max(dims)] # TODO: more dims
    assert_equal(actual, expected)


@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=2))
def test_subtract(Tx: np.ndarray):
    T, [x,y] = Tx
    f = to_python_function(op(Subtract(T)))
    assert_equal(f(x, y), [x-y])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=1))
def test_negate(Tx: np.ndarray):
    T, [x] = Tx
    f = to_python_function(op(Negate(T)))
    assert_equal(f(x), [-x])

@given(ndarrays(array_type=ndarraytypes(shape=st.just(()))))
def test_constant(Tx: np.ndarray):
    T, [x] = Tx
    f = to_python_function(op(Constant(T, x.item())))
    assert_equal(f(), [x])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=2))
def test_multiply(Tx: np.ndarray):
    T, [x, y] = Tx
    f = to_python_function(op(Multiply(T)))
    assert_equal(f(x, y), [x * y])

# >:(   np.abs(-MAXINT) = -MAXINT
def true_abs(x):
    x = np.abs(x)
    return np.where(x < 0, -(x+1), x)

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=2))
def test_power(Tx: np.ndarray):
    T, [x, y] = Tx
    # take abs of exponents; consider other values undefined.
    y = true_abs(y)
    f = to_python_function(op(Power(T)))
    assert_equal(f(x, y), [x ** y])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@pytest.mark.filterwarnings("ignore:divide by zero")
@given(ndarrays(n=2))
def test_divide(Tx: np.ndarray):
    T, [x, y] = Tx
    f = to_python_function(op(Divide(T)))
    if T.dtype.is_floating():
        expected = x / y
    else:
        expected = x // y
    assert_equal(f(x, y), [expected])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(matrix_multiply_args())
def test_matrix_multiply(NABCxy: np.ndarray):
    N, A, B, C, x, y = NABCxy
    f = to_python_function(op(MatrixMultiply(N, A, B, C)))
    assert_equal(f(x, y), [x @ y])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(composable_ndarrays())
def test_compose(ABCxy: np.ndarray):
    A, B, C, x, y = ABCxy
    f = to_python_function(op(Compose(A, B, C)))
    assert_equal(f(x, y), [np.tensordot(x, y, axes=len(B.shape))])

@given(reshape_args())
def test_reshape(XYx):
    X, Y, x = XYx
    f = to_python_function(op(Reshape(X, Y)))
    assert_equal(f(x), [x.reshape(Y.shape)])

@given(ndarrays())
def test_transpose(Tx):
    T, [x] = Tx
    p = list(reversed(range(len(T.shape))))
    f = to_python_function(op(Permute(T, p)))
    assert_equal(f(x), [x.transpose()])

@given(strategies.permute().flatmap(lambda op: st.tuples(st.just(op), ndarrays(array_type=st.just(op.T)))))
def test_permute(p_x: Tuple[Permute, np.ndarray]):
    p, (_, [x]) = p_x
    f = to_python_function(op(p))

    actual = f(x)
    expected = [x.transpose(p.p)]
    assert_equal(actual, expected)
