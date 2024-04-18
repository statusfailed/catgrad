""" Test that the Python backend behaves as expected """
import pytest
import numpy as np
from typing import List, Tuple
from hypothesis import given
from hypothesis import strategies as st

from tests.utils import assert_equal
from tests.strategies import integral_dtypes, floating_dtypes, ndarrays, ndarraytypes, composable_ndarrays, reshape_args, ncopy_args, nadd_args, nsplit_args, matrix_multiply_args
import tests.strategies as strategies

from catgrad.signature import NdArrayType, Dtype
from catgrad.target.python import to_python_function
from catgrad.target.python.array_backend import Numpy
from catgrad.bidirectional.operation import *
from catgrad.bidirectional.functor import Forget
from catgrad.compile import rdiff

F = Forget()

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=2))
def test_rd_copy(Tx: np.ndarray):
    T, [x, y] = Tx

    e = Copy(T)
    arrow = to_python_function(e.to_core())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # the fwd map is an optic with empty residual, so fwd = arrow
    assert_equal(arrow(x), [x, x])
    assert_equal(fwd(x), [x, x])
    assert_equal(rev(x, y), [x + y])

@given(ncopy_args())
def test_rd_ncopy(NxT):
    (N, [x]), T = NxT
    e = NCopy(N, T)

    arrow = to_python_function(e.to_core())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    dy = np.ones((N+T).shape) # TODO: generate
    # expected_y = np.broadcast_to(x, (N+T).shape)
    expected_y = np.broadcast_to(x.reshape(N.shape + (1,)*len(T.shape)), N.shape + T.shape)

    assert_equal(arrow(x), [expected_y])
    assert_equal(fwd(x), [expected_y])

    dims = tuple( -(i+1) for i in reversed(range(len(T.shape))) )
    assert_equal(rev(dy), [Numpy.nadd(dims, dy)])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=1))
def test_rd_discard(Tx: np.ndarray):
    T, [x] = Tx

    e = Discard(T)
    arrow = to_python_function(e.to_core())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # the fwd map is an optic with empty residual, so fwd = arrow
    assert_equal(arrow(x), [])
    assert_equal(fwd(x), [])
    assert_equal(rev(), [np.zeros(T.shape, Numpy.dtype(T.dtype))])

@given(nsplit_args())
def test_rd_nsplit(Tkx):
    (T, k, x) = Tkx
    e = NSplit(T, k)

    core = to_python_function(e.to_core())
    fwd  = to_python_function(F(e.fwd()))
    rev  = to_python_function(F(e.rev()))

    expected_y = [ x.reshape(x.shape[:-1]) for x in np.split(x, k, -1) ]
    dy = expected_y[0]

    assert_equal(core(x), expected_y)
    assert_equal(fwd(x), expected_y)
    assert_equal(rev(*expected_y), [x])

@given(nsplit_args())
def test_rd_nconcatenate(Tkx):
    (T, k, x) = Tkx
    e = NConcatenate(T, k)

    # actual inputs. Note that if list has len 1, it's already unpacked (bad choice - FIXME)
    xs = Numpy.nsplit(x, k)
    if k == 1:
        xs = [xs]

    core = to_python_function(e.to_core())
    fwd  = to_python_function(F(e.fwd()))
    rev  = to_python_function(F(e.rev()))

    expected_y = x
    dy = expected_y

    assert_equal(core(*xs), [expected_y])
    assert_equal(fwd(*xs), [expected_y])
    assert_equal(rev(dy), xs)

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=3))
def test_rd_add(Tx):
    T, [x0, x1, y] = Tx

    e = Add(T)

    arrow = to_python_function(e.to_core())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # the fwd map is an optic with empty residual, so fwd = arrow
    assert_equal(arrow(x0, x1), [x0 + x1])
    assert_equal(fwd(x0, x1), [x0 + x1])
    assert_equal(rev(y), [y, y])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(nadd_args())
def test_rd_nadd(NTx: np.ndarray):
    N, T, x = NTx
    e = NAdd(N, T)

    arrow = to_python_function(e.to_core())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    dy = np.ones((T).shape) # TODO: generate
    dims = tuple( -(i+1) for i in reversed(range(len(T.shape))) )
    expected_y = Numpy.nadd(dims, x)

    assert arrow(x)[0].shape == N.shape
    assert_equal(arrow(x), [expected_y])
    assert_equal(fwd(x), [expected_y])
    assert_equal(rev(dy), [Numpy.ncopy(T.shape, dy)])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=3))
def test_rd_subtract(Tx):
    T, [x0, x1, y] = Tx

    e = Subtract(T)

    arrow = to_python_function(e.to_core())
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

    arrow = to_python_function(e.to_core())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # the fwd map is an optic with empty residual, so fwd = arrow
    assert_equal(arrow(x), [-x])
    assert_equal(fwd(x), [-x])
    assert_equal(rev(dy), [-dy])

@given(ndarrays(n=2, array_type=ndarraytypes(dtype=integral_dtypes)))
def test_rd_invert(Tx):
    T, [x, dy] = Tx

    e = Invert(T)

    arrow = to_python_function(e.to_core())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # the fwd map is an optic with empty residual, so fwd = arrow
    assert_equal(arrow(x), [~x])
    assert_equal(fwd(x), [~x])
    assert_equal(rev(dy), [~dy])

@given(ndarrays(n=2, array_type=ndarraytypes(dtype=floating_dtypes)))
def test_rd_scale_inverse(Tx):
    T, [x, dy] = Tx
    s = 199.0 # TODO: generate a random floating point constant

    e = ScaleInverse(T, s)

    arrow = to_python_function(e.to_core())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # the fwd map is an optic with empty residual, so fwd = arrow
    s = np.array(s, dtype=x.dtype) # cast to correct dtype before comparing
    assert_equal(arrow(x), [x/s])
    assert_equal(fwd(x), [x/s])
    assert_equal(rev(dy), [dy/s])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=2, array_type=ndarraytypes(dtype=floating_dtypes)))
def test_rd_exponentiate(Tx):
    T, [x, dy] = Tx
    s = -0.5 # TODO: generate a random floating point constant

    # dodgy hack: avoid x == 0 because of division by zero
    # TODO: This kinda sucks because now we need to deal with partial functions properly!
    epsilon = 1e-06
    x = np.where(x==0, x+epsilon, x)

    e = Exponentiate(T, s)

    arrow = to_python_function(e.to_core())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # the fwd map is an optic with empty residual, so fwd = arrow
    s = np.array(s, dtype=x.dtype) # cast to correct dtype before comparing
    assert_equal(arrow(x), [x**s])
    assert_equal(fwd(x), [x**s, x])
    assert_equal(rev(x, dy), [s*x**(s-1)*dy])

@given(ndarrays(array_type=ndarraytypes(shape=st.just(()))))
def test_rd_constant(Tx: np.ndarray):
    T, [x] = Tx

    e = Constant(T, x.item())

    arrow = to_python_function(e.to_core())
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

    arrow = to_python_function(e.to_core())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # the fwd map is an optic with empty residual, so fwd = arrow
    assert_equal(arrow(x0, x1), [x0 * x1])
    assert_equal(fwd(x0, x1), [x0 * x1, x0, x1]) # this one's a lens
    assert_equal(rev(x0, x1, y), [x1 * y, x0 * y])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(matrix_multiply_args())
def test_rd_matrix_multiply(NABCxy: np.ndarray):
    N, A, B, C, x0, x1 = NABCxy

    e = MatrixMultiply(N, A, B, C)

    dy = np.ones((N+A+C).shape, dtype=Numpy.dtype(N.dtype))

    core = to_python_function(e.to_core())
    fwd  = to_python_function(F(e.fwd()))
    rev  = to_python_function(F(e.rev()))

    assert_equal(core(x0, x1), [x0 @ x1])
    assert_equal(fwd(x0, x1), [x0 @ x1, x0, x1]) # lens
    n = len(N.shape)
    p = list(range(n)) + [n+1, n]
    assert_equal(rev(x0, x1, dy), [dy @ x1.transpose(p), x0.transpose(p) @ dy])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(composable_ndarrays())
def test_rd_compose(ABCxy: np.ndarray):
    A, B, C, x0, x1 = ABCxy
    # TODO: generate a random dy
    dy = np.ones((A+C).shape, Numpy.dtype(A.dtype))
    e = Compose(A, B, C)

    arrow = to_python_function(e.to_core())
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

    arrow = to_python_function(e.to_core())
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

    arrow = to_python_function(e.to_core())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    expected_y = x.transpose(e.p)

    assert_equal(arrow(x), [expected_y])
    assert_equal(fwd(x), [expected_y]) # this one's a lens
    assert_equal(rev(expected_y), [x]) # inverse

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=3))
def test_rd_gt(Tx):
    T, [x0, x1, y] = Tx

    e = Gt(T)

    arrow = to_python_function(e.to_core())
    fwd   = to_python_function(F(e.fwd()))
    rev   = to_python_function(F(e.rev()))

    # the fwd map is an optic with empty residual, so fwd = arrow
    assert_equal(arrow(x0, x1), [x0 > x1])
    assert_equal(fwd(x0, x1), [x0 > x1])
    z = np.zeros(T.shape, Numpy.dtype(T.dtype))
    assert_equal(rev(y), [z, z])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=2, array_type=ndarraytypes(dtype=st.just(Dtype.float32))))
def test_rd_sigmoid(Tx):
    T, [x, dy] = Tx

    e = Sigmoid(NdArrayType((), Dtype.float32))

    arrow = to_python_function(e.to_core())
    fwd = to_python_function(F(e.fwd()))
    rev = to_python_function(F(e.rev()))

    from scipy.special import expit
    rexpit = lambda x, dy: (expit(x) * (1 - expit(x)) * dy).astype(Numpy.dtype(T.dtype))

    assert_equal(arrow(x), [expit(x)], exact=False)
    # NOTE: Sigmoid has a custom fwd map, where the *sigmoid* of the input is sent to the rev. map.
    # This is so we don't have to compute it twice.
    assert_equal(fwd(x), [expit(x), expit(x)], exact=False)
    assert_equal(rev(*arrow(x), dy), [rexpit(x, dy)], exact=False)

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=2, array_type=ndarraytypes(dtype=st.just(Dtype.float32))))
def test_rd_relu(Tx):
    T, [x, dy] = Tx

    f = relu(obj(T))
    r = rdiff(f)

    fwd = to_python_function(F(f))
    rev = to_python_function(F(r))
    assert_equal(fwd(x), [(x>0)*x], exact=True)
    assert_equal(rev(x, dy), [(x>0)*dy], exact=True)
