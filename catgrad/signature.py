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

def obj(*args: List[NdArrayType]) -> FiniteFunction:
    """ Create an object of the category (a FiniteFunction ``X → Σ₀``) from a
    list of ``NdArrayType`` """
    table = FiniteFunction.Array.array(args, OBJ_DTYPE)
    return FiniteFunction(None, table)

################################################################################
# Operations

class Operation(ABC):
    # TODO: replace this with source/target?
    @property
    @abstractmethod
    def op(self) -> OpenHypergraph:
        ...

# Turn a single operation into an OpenHypergraph
def op(x: Operation, A: FiniteFunction, B: FiniteFunction):
    x = FiniteFunction(None, FiniteFunction.Array.array([x], 'O'))
    return OpenHypergraph.singleton(x, A, B)
