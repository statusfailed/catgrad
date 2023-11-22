from typing import Any
from dataclasses import dataclass
from open_hypergraphs import OpenHypergraph

from catgrad.signature import NdArrayType, obj, op, Operation

def prod(xs):
    a = 1
    for x in xs:
        a *= x
    return a

################################################################################
# Cartesian Left-Additive Structure

@dataclass
class Copy(Operation):
    T: NdArrayType

    @property
    def op(self) -> OpenHypergraph:
        return op(self, obj(self.T), obj(self.T, self.T))

@dataclass
class Discard(Operation):
    T: NdArrayType

    @property
    def op(self) -> OpenHypergraph:
        return op(self, obj(self.T), obj())

@dataclass
class Add(Operation):
    T: NdArrayType

    @property
    def op(self) -> OpenHypergraph:
        return op(self, obj(self.T, self.T), obj(self.T))

@dataclass
class Constant(Operation):
    T: NdArrayType
    x: Any # will be cast to T.dtype

    def __post_init__(self):
        if self.T.shape != ():
            raise ValueError(f"Constant.T.shape must be () but was {self.T.dtype}")

    @property
    def op(self) -> OpenHypergraph:
        return op(self, obj(), obj(self.T))

################################################################################
# Isomorphisms

@dataclass
class Reshape(Operation):
    X: NdArrayType
    Y: NdArrayType

    def __post_init__(self):
        # input and output must have same number of entries
        if prod(self.X.shape) != prod(self.Y.shape):
            raise ValueError("Must have prod(X) == prod(Y)")

    @property
    def op(self) -> OpenHypergraph:
        return op(self, obj(self.X), obj(self.Y))

# TODO: Replace this with explicit swap two axes rather than reversing all of them.
@dataclass
class Transpose(Operation):
    """ Reverse the dimensions of a tensor """
    T: NdArrayType

    @property
    def target(self):
        return NdArrayType(tuple(reversed(self.T.shape)), self.T.dtype)

    @property
    def op(self) -> OpenHypergraph:
        return op(self, obj(self.T), obj(self.target))

################################################################################
# Cartesian Distributive Structure

@dataclass
class Multiply(Operation):
    T: NdArrayType

    @property
    def op(self) -> OpenHypergraph:
        return op(self, obj(self.T, self.T), obj(self.T))

################################################################################
# Matrix multiplication

@dataclass
class Compose(Operation):
    """ Composition of tensors ``f : A → B`` and ``g : B → C`` along ``B``, so
    that ``Compose(f, g) : A → C``
    """
    A: NdArrayType
    B: NdArrayType
    C: NdArrayType

    @property
    def op(self) -> OpenHypergraph:
        A, B, C = self.A, self.B, self.C
        S = obj(A+B, B+C)
        T = obj(A+C)
        return op(self, S, T)
