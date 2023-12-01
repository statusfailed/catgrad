from typing import List, Tuple, Any, Callable, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from open_hypergraphs import FiniteFunction, OpenHypergraph

# Initial maps into Σ₀ and Σ₁
OP_DTYPE = object
OBJ_DTYPE = object
sigma_0 = FiniteFunction.initial(None, dtype=OBJ_DTYPE)
sigma_1 = FiniteFunction.initial(None, dtype=OP_DTYPE)

################################################################################
# Objects of the category

Shape = Tuple

class Dtype(Enum):
    """ Dtypes correspond to semirings: a choice of dtype specifies field/(semi)ring operations. """
    int32 = auto()
    float32 = auto()

    def is_floating(self):
        return self.value in _FLOATING_DTYPES
_FLOATING_DTYPES = { Dtype.float32.value }

@dataclass
class NdArrayType:
    """ An NdArrayType is the *metadata* for an N-dimensional array.
    It consists of a shape and a dtype, but not data.
    The values of type ``NdArrayType`` are the generating objects of our category.
    """
    shape: Shape
    dtype: Dtype

    @classmethod
    def from_ndarray(cls, x: np.ndarray):
        """ Create an ``NdArrayType`` from an object with shape and dtype """
        return NdArrayType(x.shape, x.dtype)

    # coproduct of shapes for convenience!
    # TODO: should this be __mul__? It's the cartesian product!
    def __add__(x, y):
        """ Take the coproduct of two objects """
        # allow e.g., NdArrayType((1,2,3), int) + 4 = NdArrayType((1,2,3,4), int)
        # and 0 + NdArrayType((1,2,3), int) = NdArrayType((0,1,2,3), int)
        if type(x) is NdArrayType and type(y) is int:
            return x + NdArrayType((y,), x.dtype)
        if type(y) is int and type(x) is NdArrayType:
            return x + NdArrayType((y,), x.dtype)

        if x.dtype != y.dtype:
            raise ValueError(f"Can't concatenate {x} and {y} with differing dtypes")

        return NdArrayType(x.shape + y.shape, x.dtype)

def obj(*args) -> FiniteFunction:
    """ Create an object of the category (a FiniteFunction ``X → Σ₀``) from a
    list of ``NdArrayType`` """
    table = FiniteFunction.Array.array(args, OBJ_DTYPE)
    return FiniteFunction(None, table)

################################################################################
# TODO?

# Turn a single operation into an OpenHypergraph
def singleton(x: Any, A: FiniteFunction, B: FiniteFunction):
    f = FiniteFunction(None, FiniteFunction.Array.array([x], 'O'))
    return OpenHypergraph.singleton(f, A, B)

class Typed(Protocol):
    def source(self) -> FiniteFunction:
        ...
    def target(self) -> FiniteFunction:
        ...

def op(x: Typed):
    return singleton(x, x.source(), x.target())
