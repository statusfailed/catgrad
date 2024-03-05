from hypothesis import given
import numpy as np

from catgrad.combinators import *
from catgrad.signature import *
from catgrad.parameters import *
from catgrad.bidirectional.operation import *
from catgrad.target.python import to_python_function
from catgrad.target.python.array_backend import Numpy

from tests.strategies import ndarraytypes, composable_ndarraytypes, objects
from tests.utils import assert_open_hypergraphs_equal

# TODO: surprisingly, we get strict equality usefully often here.
#       This check is quite brittle to specific choice of representative, so it
#       should be replaced by proper isomorphism checking
# TODO: test `factor` properly: it's more general than just factoring out parameters!

@given(ndarraytypes())
def test_parameter1(T: NdArrayType):
    c = op(Parameter(T))
    e, d = factor_parameters(c)
    assert c == (e @ identity(c.source)) >> d

@given(ndarraytypes())
def test_bias1(T: NdArrayType):
    c = (op(Parameter(T)) @ identity(obj(T))) >> add(obj(T))
    e, d = factor_parameters(c)
    assert c == (e @ identity(c.source)) >> d

@given(objects())
def test_bias(A: FiniteFunction):
    c = (parameter(A) @ identity(A)) >> add(A)
    e, d = factor_parameters(c)
    assert c == (e @ identity(c.source)) >> d

@given(composable_ndarraytypes())
def test_dense(ABC: Tuple[NdArrayType, NdArrayType, NdArrayType]):
    A, B, C = ABC
    # manually construct a dense layer
    linear = (identity(obj(A+B)) @ parameter(obj(B+C))) >> op(Compose(A,B,C))
    bias = (parameter(obj(A+C)) @ identity(obj(A+C))) >> add(obj(A+C))
    c = linear >> bias
    e, d = factor_parameters(c)
    
    # this one's not equal on the nose, so we can only check some invariants
    # without finding an explicit isomorphism.
    assert_open_hypergraphs_equal(c, (e @ identity(c.source)) >> d)

    # make sure d compiles to something which actually computes like a dense layer
    F = Forget()
    f = to_python_function(F(d))

    n_AB = np.prod((A+B).shape)
    n_BC = np.prod((B+C).shape)
    n_AC = np.prod((A+C).shape)

    # use 0,1,2...N as our test data
    x = np.arange(n_AB).reshape((A+B).shape)
    P_linear = np.arange(n_BC).reshape((B+C).shape)
    P_bias = np.arange(n_AC).reshape((A+C).shape)

    assert np.all(f(P_linear, P_bias, x) == (Numpy.compose(x, P_linear, len(B.shape)) + P_bias))
