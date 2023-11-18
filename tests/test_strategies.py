# are strategies are generating what we expect?
from hypothesis import given
from tests.strategies import *

@given(shapes())
def test_shapes(s):
    # make sure arrays not too big!
    p = 1
    for dim in s:
        p *= dim
    assert p <= MAX_ELEMENTS
