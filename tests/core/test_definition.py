from catgrad.signature import Dtype, op
from catgrad.core.definition import Sigmoid
from catgrad.core.operation import Add, Negate

from catgrad.special.definition import inline
from catgrad.target.python import to_python_function

# reference impl.
from scipy.special import expit

# test libs
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.utils import assert_equal
from tests.strategies import ndarrays, ndarraytypes
import tests.strategies as strategies

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=1, array_type=ndarraytypes(dtype=st.just(Dtype.float32))))
def test_inline_sigmoid(Tx):
    """ check we can compile an inlined catgrad.core.definition.Sigmoid """
    T, [x] = Tx
    c = op(Sigmoid(T))
    f = to_python_function(inline(c))

    actual = f(x)
    expected = [expit(x)]
    assert_equal(actual, expected, exact=False)

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=1, array_type=ndarraytypes(dtype=st.just(Dtype.float32))))
def test_definition_sigmoid(Tx):
    """ check we can compile an inlined catgrad.core.definition.Sigmoid """
    T, [x] = Tx
    c = op(Sigmoid(T))

    f = to_python_function(c)

    actual = f(x)
    expected = [expit(x)]
    assert_equal(actual, expected, exact=False)

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=2, array_type=ndarraytypes(dtype=st.just(Dtype.float32))))
def test_definition_sigmoid_complex(Tx):
    """ a more complex example involving ``catgrad.core.definition.Sigmoid`` """
    T, [x, y] = Tx
    c = op(Add(T)) >> op(Sigmoid(T)) >> op(Negate(T))

    f = to_python_function(c)

    actual = f(x, y)
    expected = [-expit(x+y)]
    assert_equal(actual, expected, exact=False)
