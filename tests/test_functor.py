import numpy as np
import pytest
from hypothesis import given, example

from tests.utils import assert_equal
import tests.strategies as strategies

from open_hypergraphs import FiniteFunction, OpenHypergraph

from catgrad.target.python import to_python_function, to_python_class_ast
from catgrad.target.python.array_backend import Numpy
from catgrad.signature import op, obj, NdArrayType, Dtype
from catgrad.rdops import Forget, Compose, copy, discard, add, zero, multiply, constant, negate
from catgrad.combinators import identity

from catgrad.functor import Bidirectional

F = Forget()
B = Bidirectional()

# NOTE: we only test a few different circuits in this file; we're testing the Bidirectional functor, *not* the circuits.
@example(obj(NdArrayType((), Dtype.int32), NdArrayType((), Dtype.float32)))
@given(strategies.objects())
def test_bidirectional_zero(A: FiniteFunction):
    c = zero(A)
    Bc = B(c)

    assert Bc.source == FiniteFunction.transpose(1, len(c.source)) >> (c.source + c.source)
    assert Bc.target == FiniteFunction.transpose(2, len(c.target)) >> (c.target + c.target)

    d = F(B.adapt(Bc, c.source, c.target))
    f = to_python_function(d)

    # TODO: get random args from hypothesis
    args = [ np.ones(shape=T.shape, dtype=Numpy.dtype(T.dtype)) for T in A ]
    expected = [ np.zeros(shape=T.shape, dtype=Numpy.dtype(T.dtype)) for T in c.target ]
    actual = f(*args)
    assert_equal(expected, actual)

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(strategies.objects_and_values(add))
def test_bidirectional_add(Xcv):
    X, c, v = Xcv
    Bc = B(c)

    d = F(B.adapt(Bc, c.source, c.target))
    f = to_python_function(d)

    args = v
    xs = v[:len(X)]
    ys = v[len(X):]
    dy = [ np.ones(shape=A.shape, dtype=Numpy.dtype(A.dtype)) for A in X ]

    fwd_expected = [ x + y for x, y in zip(xs, ys) ]
    rev_expected = to_python_function(F(copy(c.target)))(*dy)

    expected = fwd_expected + rev_expected
    actual = f(*args, *dy)
    assert_equal(expected, actual)

@given(strategies.objects_and_values(copy))
def test_bidirectional_copy(Xcv):
    X, c, v = Xcv
    Bc = B(c)

    d = F(B.adapt(Bc, c.source, c.target))
    f = to_python_function(d)

    args = v
    dy = [ np.ones(shape=A.shape, dtype=Numpy.dtype(A.dtype)) for A in X ]
    dy = dy + dy

    fwd_expected = args + args
    rev_expected = to_python_function(F(add(c.source)))(*dy)

    expected = fwd_expected + rev_expected
    actual = f(*args, *dy)
    assert_equal(expected, actual)

@given(strategies.objects_and_values(negate))
def test_double_negate(Xcv):
    X, c, v = Xcv
    c = (c >> c) @ (c >> c)
    d = F(B.adapt(B(c), c.source, c.target))
    f = to_python_function(d)

    args = v
    dy = [ np.ones(shape=A.shape, dtype=Numpy.dtype(A.dtype)) for A in X ]

    # double negation is identity
    assert_equal(f(*v, *v, *dy, *dy), [*v, *v, *dy, *dy])

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(strategies.ndarrays(n=4))
def test_mul_assoc(Tx):
    T, xs = Tx
    # keep arrays small, otherwise we end up with unfortunate FP-non-associative-weirdness.
    a_min = Numpy.dtype(T.dtype)(-2**15)
    a_max = Numpy.dtype(T.dtype)(2**15)
    x0, x1, x2, dy = [ np.clip(v, a_min, a_max) for v in xs ]

    T = obj(T)

    c = (multiply(T) @ identity(T)) >> multiply(T)
    d = F(B.adapt(B(c), c.source, c.target))
    f = to_python_function(d)

    actual = f(x0, x1, x2, dy)
    expected = [x0 * x1 * x2, dy*x1*x2, dy*x0*x2, dy*x0*x1]
    assert_equal(actual, expected, exact=False)

def test_compose_assoc():
    # This testcase specifically reproduces a (now fixed) bug in
    # open_hypergraphs where "interleave_blocks" maps were incorrectly inverted.
    T = NdArrayType((1,), Dtype.int32)
    U = NdArrayType((2,), Dtype.int32)
    V = NdArrayType((3,), Dtype.int32)
    W = NdArrayType((4,), Dtype.int32)

    c = (op(Compose(T,U,V)) @ identity(obj(V+W))) >> op(Compose(T,V,W))

    x0 = np.ones((1,2), np.int32)
    x1 = np.ones((2,3), np.int32)
    x2 = np.ones((3,4), np.int32)

    dy = np.ones((1,4), np.int32)

    d = F(B.adapt(B(c), c.source, c.target))
