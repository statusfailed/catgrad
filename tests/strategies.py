import math
import operator
from functools import reduce

import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

from catgrad.signature import Dtype, NdArrayType
from catgrad.target.python.array_backend import Numpy

dtypes = st.sampled_from(Dtype)

MAX_ELEMENTS = 1_000_000
@st.composite
def shapes(draw, max_elements=st.just(MAX_ELEMENTS)):
    # draw shapes with total number of elements not exceeding 1e6
    max_elements = draw(max_elements)
    ndim = draw(st.integers(min_value=0, max_value=6))
    max_dim = math.floor(max_elements**(1/ndim)) if ndim > 0 else max_elements
    dimensions = st.integers(min_value=0, max_value=max_dim)
    result = draw(st.lists(elements=dimensions, min_size=ndim, max_size=ndim))
    return tuple(result)

@st.composite
def ndarraytypes(draw, shape=shapes(), dtype=dtypes):
    dtype = draw(dtype)
    shape = draw(shape)
    return NdArrayType(shape, dtype)

@st.composite
def ndarrays(draw, array_type=ndarraytypes(), n=1):
    t = draw(array_type)
    xs = draw(st.lists(arrays(shape=t.shape, dtype=Numpy.dtype(t.dtype)), min_size=n, max_size=n))
    return t, xs

@st.composite
def composable_ndarrays(draw):
    # we have to make pretty small arrays to fit in the test deadline
    shape = shapes(max_elements=st.just(500))
    A = draw(ndarraytypes(shape=shape))
    B = draw(ndarraytypes(dtype=st.just(A.dtype), shape=shape))
    C = draw(ndarraytypes(dtype=st.just(A.dtype), shape=shape))

    _, [x] = draw(ndarrays(array_type=st.just(A+B), n=1))
    _, [y] = draw(ndarrays(array_type=st.just(B+C), n=1))
    return A, B, C, x, y

@st.composite
def reshape_args(draw):
    T0 = draw(ndarraytypes(shape=shapes(max_elements=st.just(1000))))
    T1 = draw(ndarraytypes(shape=shapes(max_elements=st.just(1000))))

    p0 = reduce(operator.mul, T0.shape, 1)
    p1 = reduce(operator.mul, T1.shape, 1)

    # since this is the cartesian product, this ensures the total number of
    # elements is equal.
    T0 = T0 + p1
    T1 = T1 + p0

    _, [x] = draw(ndarrays(array_type=st.just(T0), n=1))
    return T0, T1, x
