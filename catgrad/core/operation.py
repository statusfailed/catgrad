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
    ``NCopy(N,T) : N → N+T`` is like the T-fold copy of a tensor of shape N, then packed into a tensor.
    """
    N: NdArrayType
    T: NdArrayType
    def source(self): return obj(self.N)
    def target(self): return obj(self.N + self.T)

@dataclass
class Discard:
    T: NdArrayType
    def source(self): return obj(self.T)
    def target(self): return obj()

@dataclass
class NSplit:
    """ ``NSplit(T, k) : T*k → T●T..k...T`` splits a tensor into individual outputs """
    T: NdArrayType
    k: int
    def __post_init__(self):
        # TODO: this should really be equivalent to discarding.
        assert self.k > 0, "cannot split into fewer than 1 outputs"

    def source(self):
        N = NdArrayType((self.k,), self.T.dtype)
        return obj(self.T + N)
    def target(self): return obj(*([self.T]*self.k)) # N●..k..●N

@dataclass
class NConcatenate:
    """ ``NConcatenate(N, k) : N●N..k...N → N*k`` concatenates k tensors into a single outputs """
    T: NdArrayType
    k: int
    def source(self): return obj(*([self.T]*self.k)) # N●..k..●N
    def target(self):
        N = NdArrayType((self.k,), self.T.dtype)
        return obj(self.T + N)

@dataclass
class Add:
    T: NdArrayType
    def source(self): return obj(self.T, self.T)
    def target(self): return obj(self.T)

@dataclass
class NAdd:
    """ ``NAdd(N, T) : N×T → N`` sums a tensor of type N + T over the T dimensions. """
    N: NdArrayType
    T: NdArrayType
    def source(self): return obj(self.N + self.T)
    def target(self): return obj(self.N)

@dataclass
class NMax:
    """ ``NMax(N, T) : N×T → N`` returns the maximum of a tensor of type N + T over the T dimensions. """
    N: NdArrayType
    T: NdArrayType
    def __post_init__(self):
        if prod(self.N.shape + self.T.shape) <= 0:
            raise ValueError(f"Cannot NMax zero-element vector: {self}")

    def source(self): return obj(self.N + self.T)
    def target(self): return obj(self.N)

@dataclass
class Negate:
    """ ``Negate(T) : T×T → T`` computes ``-x``. """
    T: NdArrayType
    def source(self): return obj(self.T)
    def target(self): return obj(self.T)

@dataclass
class Invert:
    T: NdArrayType
    def __post_init__(self):
        if self.T.dtype.is_floating():
            raise ValueError("Invert operation not supported for floating dtypes")
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
# Comparators

@dataclass
class Gt:
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
    return (op(Constant(T, math.e)) @ identity(obj(T))) >> op(Power(T))

################################################################################
# Matrix multiplication and tensor composition

@dataclass
class MatrixMultiply:
    """ Batched matrix multiplication """
    N: NdArrayType
    A: NdArrayType
    B: NdArrayType
    C: NdArrayType
    def __post_init__(self):
        # N can be any type, but A, B, C must be a single dimension
        for t in [self.A, self.B, self.C]:
            assert len(t.shape) == 1

    def source(self): return obj(self.N+self.A+self.B, self.N+self.B+self.C)
    def target(self): return obj(self.N+self.A+self.C)

# TODO: should we get rid of this?
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

operation = Copy | NCopy | NSplit | NConcatenate | Discard | Add | NAdd | Constant | Reshape | Permute | Multiply | MatrixMultiply | Compose
