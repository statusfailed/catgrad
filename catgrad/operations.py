from typing import Any, List
import math
from dataclasses import dataclass
from open_hypergraphs import OpenHypergraph

from catgrad.signature import NdArrayType, obj, op
from catgrad.combinators import identity

def prod(xs):
    a = 1
    for x in xs:
        a *= x
    return a

################################################################################
# Cartesian Left-Additive Structure

@dataclass
class Copy:
    T: NdArrayType
    def source(self): return obj(self.T)
    def target(self): return obj(self.T, self.T)

@dataclass
class NCopy:
    # TODO: introduce split/join and mention naturality of Broadcast w.r.t. them.
    """ NCopy is like Copy, but on tensor dimensions.
    ``NCopy(N,A) : A → N+A`` is like the N-fold copy of a tensor of shape A, then packed into a tensor.
    """
    N: NdArrayType
    T: NdArrayType
    def source(self): return obj(self.T)
    def target(self): return obj(self.N + self.T)

@dataclass
class Discard:
    T: NdArrayType
    def source(self): return obj(self.T)
    def target(self): return obj()

@dataclass
class Add:
    T: NdArrayType
    def source(self): return obj(self.T, self.T)
    def target(self): return obj(self.T)

@dataclass
class NAdd:
    """ ``NAdd(N, T)`` sums a tensor of type N + T over the N dimensions. """
    N: NdArrayType
    T: NdArrayType
    def source(self): return obj(self.N + self.T)
    def target(self): return obj(self.T)

@dataclass
class Negate:
    """ ``Negate(T) : T×T → T`` computes ``-x``. """
    T: NdArrayType
    def source(self): return obj(self.T)
    def target(self): return obj(self.T)

@dataclass
class Subtract:
    """ ``Sub(T) : T×T → T`` computes ``(x - y)`` """
    T: NdArrayType
    def source(self): return obj(self.T, self.T)
    def target(self): return obj(self.T)

scalar = int | float
@dataclass
class Constant:
    T: NdArrayType
    x: scalar # will be cast to T.dtype

    def __post_init__(self):
        if not isinstance(self.x, scalar):
            raise ValueError(f"constant {self.x} is not a scalar {scalar}")
        if self.T.shape != ():
            raise ValueError(f"Constant.T.shape must be () but was {self.T.shape}")

    def source(self): return obj()
    def target(self): return obj(self.T)

################################################################################
# Isomorphisms

@dataclass
class Reshape:
    X: NdArrayType
    Y: NdArrayType

    def source(self): return obj(self.X)
    def target(self): return obj(self.Y)

    def __post_init__(self):
        # input and output must have same number of entries
        if prod(self.X.shape) != prod(self.Y.shape):
            raise ValueError("Must have prod(X) == prod(Y)")

# TODO: Replace this with explicit swap two axes rather than reversing all of them.
@dataclass
class Permute:
    """ Permute the dimensions of a tensor """
    T: NdArrayType
    p: List[int]

    def __post_init__(self):
        k = len(self.T.shape)
        if list(sorted(self.p)) != list(range(k)):
            raise ValueError(f"p must be a permutation of type {k} → {k}")

    def source(self): return obj(self.T)
    def target(self):
        U_shape = tuple(self.T.shape[i] for i in self.p)
        U = NdArrayType(U_shape, self.T.dtype)
        return obj(U)

################################################################################
# Cartesian Distributive Structure

@dataclass
class Multiply:
    T: NdArrayType
    def source(self): return obj(self.T, self.T)
    def target(self): return obj(self.T)

################################################################################
# Partial functions arithmetic

@dataclass
class Divide:
    """ `Divide: T × T → T` is the partial function `<x,y> → x/y`. """
    T: NdArrayType
    def source(self): return obj(self.T, self.T)
    def target(self): return obj(self.T)

@dataclass
class Power:
    """ `Power: T × T → T` is the partial function `<x,y> → x^y`. """
    T: NdArrayType
    def source(self): return obj(self.T, self.T)
    def target(self): return obj(self.T)

# TODO: tidy this up
# helper for the 'exp' circuit.
def exp1(T: NdArrayType):
    U = NdArrayType((), T.dtype)
    a = op(Constant(U, math.e))
    b = op(NCopy(T, U))
    return ((a >> b) @ identity(obj(T))) >> op(Power(T))

################################################################################
# Matrix multiplication

@dataclass
class Compose:
    """ Composition of tensors ``f : A → B`` and ``g : B → C`` along ``B``, so
    that ``Compose(f, g) : A → C``
    """
    A: NdArrayType
    B: NdArrayType
    C: NdArrayType

    def source(self): return obj(self.A+self.B, self.B+self.C)
    def target(self): return obj(self.A+self.C)

################################################################################
# All the array operations in a union type

operation = Copy | NCopy | Discard | Add | NAdd | Constant | Reshape | Permute | Multiply | Compose
