from dataclasses import dataclass

from catgrad.signature import NdArrayType, Dtype, op, obj
from catgrad.target.python import to_python_function, to_python_class_ast

# used in tests
from catgrad.core.operation import Negate
from catgrad.combinators import identity

# SUT
from catgrad.target.python.special import PythonOp, IdentityPythonOp

# reference impl.
from scipy.special import expit

# test libs
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.utils import assert_equal
from tests.strategies import ndarrays, ndarraytypes

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=1, array_type=ndarraytypes(dtype=st.just(Dtype.float32))))
def test_python_op_identity(Tx):
    """ Check the Identity PythonOp actually compiles and runs """
    T, [x] = Tx
    f = to_python_function(op(IdentityPythonOp(T)))

    actual = f(x)
    expected = [x]
    assert_equal(actual, expected, exact=True)

# A 2 → 2 operation for testing multiple input/output ops
@dataclass(frozen=True)
class AddSubOp(PythonOp):
    T: NdArrayType
    def source(self): return obj(self.T, self.T)
    def target(self): return obj(self.T, self.T)
    def __call__(self, x, y):
        return [x+y, x-y]

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:invalid value")
@given(ndarrays(n=2, array_type=ndarraytypes(dtype=st.just(Dtype.float32))))
def test_python_op_addsub(Txy):
    """ Check the Identity PythonOp actually compiles and runs """
    T, [x, y] = Txy
    # make a nontrivial circuit representing the expressions
    #   [ (-x) + y, -x-y ]
    c = (op(Negate(T)) @ identity(obj(T))) >> op(AddSubOp(T))
    f = to_python_function(c)

    actual = f(x, y)
    expected = [(-x)+y, (-x) - y]
    assert_equal(actual, expected, exact=True)

def test_to_python_class_ast_errors_with_python_op():
    T = NdArrayType((2,3,4), Dtype.float32)
    c = op(IdentityPythonOp(T))
    with pytest.raises(Exception):
        to_python_class_ast(c)
