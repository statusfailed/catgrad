import pytest
import numpy as np
from hypothesis import given

from tests.utils import assert_equal
import tests.strategies as strategies

from catgrad.target.python import to_python_function
from catgrad.target.python.array_backend import Numpy
from catgrad.bidirectional.operation import copy, discard, add, zero, multiply, constant
from catgrad.bidirectional.functor import Forget

F = Forget()

from hypothesis import settings, reproduce_failure
@settings(print_blob=True)
@given(strategies.objects_and_values(copy))
def test_object_copy(Xcv):
    X, c, v = Xcv

    # smoketest for objects_and_values generator
    assert c.source == X
    assert c.target == X + X

    f = to_python_function(F(c))
    expected = v+v
    actual = f(*v)
    assert_equal(expected, actual)

@given(strategies.objects_and_values(discard))
def test_object_discard(Xcv):
    X, c, v = Xcv
    assert len(c.target) == 0
    f = to_python_function(F(c))

    expected = []
    actual = f(*v)
    assert_equal(expected, actual)

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(strategies.objects_and_values(add))
def test_object_add(Xcv):
    X, c, v = Xcv

    # smoketest for objects_and_values generator
    assert len(v) % 2 == 0
    assert c.source == X + X
    assert c.target == X

    f = to_python_function(F(c))

    xs = v[:len(X)]
    ys = v[len(X):]
    expected = [ x + y for x, y in zip(xs, ys) ]
    actual = f(*v)
    assert_equal(expected, actual)


@given(strategies.objects_and_values(zero))
def test_object_zero(Xcv):
    X, c, v = Xcv
    assert len(v) == 0 # smoketest for objects_and_values generator
    f = to_python_function(F(c))
    expected = [ np.zeros(shape=T.shape, dtype=Numpy.dtype(T.dtype)) for T in c.target ]
    actual = f(*v)
    assert_equal(expected, actual)

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(strategies.objects_and_values(multiply))
def test_object_add(Xcv):
    X, c, v = Xcv

    # smoketest for objects_and_values generator
    assert len(v) % 2 == 0
    assert c.source == X + X
    assert c.target == X

    f = to_python_function(F(c))

    xs = v[:len(X)]
    ys = v[len(X):]
    expected = [ x * y for x, y in zip(xs, ys) ]
    actual = f(*v)
    assert_equal(expected, actual)


@given(strategies.objects_and_values(constant(1)))
def test_object_constant_1(Xcv):
    X, c, v = Xcv
    assert len(v) == 0 # smoketest for objects_and_values generator
    f = to_python_function(F(c))
    expected = [ np.ones(shape=T.shape, dtype=Numpy.dtype(T.dtype)) for T in c.target ]
    actual = f(*v)
    assert_equal(expected, actual)
