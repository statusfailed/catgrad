import numpy as np
from dataclasses import dataclass
from abc import abstractmethod
from open_hypergraphs import OpenHypergraph, FiniteFunction, IndexedCoproduct, FrobeniusFunctor

from catgrad.signature import NdArrayType, obj, op, sigma_0, sigma_1
import catgrad.core.operation as ops
import catgrad.core.definition as defs
from catgrad.combinators import *

class Optic:
    @abstractmethod
    def source(self) -> FiniteFunction:
        ...

    @abstractmethod
    def target(self) -> FiniteFunction:
        ...

    # Map this Optic into its underlying array operations
    # e.g., Sigmoid will map to exp / (1 + exp)
    @abstractmethod
    def to_core(self) -> OpenHypergraph:
        ...

    # Optic structure
    @abstractmethod
    def fwd(self) -> OpenHypergraph:
        ...

    @abstractmethod
    def rev(self) -> OpenHypergraph:
        ...

    @abstractmethod
    def residual(self) -> FiniteFunction:
        ...


# Helper for lenses
class Lens(Optic):
    """ Lenses are optics whose forward map has the form ``Δ ; (f × id)`` """
    def fwd(self) -> OpenHypergraph:
        A = self.source()
        return copy(A) >> (op(self) @ identity(A))

    def residual(self) -> FiniteFunction:
        return self.source()

# Helper for linear maps (linear in the reverse-derivative sense!)
class Dagger(Optic):
    """ Dagger optics have unit residuals and a dagger reverse map """
    @abstractmethod
    def dagger(self) -> OpenHypergraph:
        ...

    def fwd(self) -> OpenHypergraph:
        return op(self)

    def rev(self) -> OpenHypergraph:
        return self.dagger()

    def residual(self) -> FiniteFunction:
        return obj()

################################################################################
# basic operations as optics

class Copy(ops.Copy, Dagger):
    def to_core(self): return op(ops.Copy(self.T))
    def dagger(self): return op(Add(self.T))

class NCopy(ops.NCopy, Dagger):
    def to_core(self): return op(ops.NCopy(self.N, self.T))
    def dagger(self): return op(NAdd(self.N, self.T))

class Discard(ops.Discard, Dagger):
    def to_core(self): return op(ops.Discard(self.T))
    def dagger(self):
        return op(Constant(self.T, 0))

class Add(ops.Add, Dagger):
    def to_core(self): return op(ops.Add(self.T))
    def dagger(self): return op(Copy(self.T))

class NAdd(ops.NAdd, Dagger):
    def to_core(self): return op(ops.NAdd(self.N, self.T))
    def dagger(self): return op(NCopy(self.N, self.T))

class Subtract(ops.Add, Dagger):
    def to_core(self): return op(ops.Subtract(self.T))
    def dagger(self):
        T = obj(self.T)
        return copy(T) >> (identity(T) @ negate(T))

class Negate(ops.Negate, Dagger):
    def to_core(self): return op(ops.Negate(self.T))
    def dagger(self): return op(self)

class Invert(ops.Invert, Dagger):
    def to_core(self): return op(ops.Invert(self.T))
    def dagger(self): return op(self)

class Constant(ops.Constant, Dagger):
    def to_core(self): return op(ops.Constant(self.T, self.x))
    def dagger(self): return op(Discard(self.T))

class Reshape(ops.Reshape, Dagger):
    def to_core(self): return op(ops.Reshape(self.X, self.Y))
    def dagger(self): return op(Reshape(self.Y, self.X))

class Permute(ops.Permute, Dagger):
    def to_core(self): return op(ops.Permute(self.T, self.p))
    def dagger(self): return op(Permute(self.T, np.argsort(self.p).tolist()))

class Gt(ops.Gt, Optic):
    def to_core(self): return op(ops.Gt(self.T))
    def residual(self): return obj()
    def fwd(self): return op(self)
    def rev(self):
        return op(Discard(self.T)) >> zero(obj(self.T, self.T))

class Multiply(ops.Multiply, Lens):
    def to_core(self): return op(ops.Multiply(self.T))
    def rev(self):
        T = obj(self.T)
        mul = op(self)
        lhs = (twist(T, T) @ copy(T))
        mid = (identity(T) @ twist(T, T) @ identity(T))
        rhs = mul @ mul
        return lhs >> mid >> rhs

class Compose(ops.Compose, Lens):
    """ Tensor composition (diagrammatic order) """
    def to_core(self): return op(ops.Compose(self.A, self.B, self.C))
    def rev(self):
        A, B, C = self.A, self.B, self.C
        t_AB = FiniteFunction.twist(len(B.shape), len(A.shape)).table.tolist()
        t_BC = FiniteFunction.twist(len(C.shape), len(B.shape)).table.tolist()

        lhs = op(Permute(A+B, t_AB)) @ op(Permute(B+C, t_BC)) @ copy(obj(A + C))
        p = permutation(lhs.target, [2, 1, 0, 3])
        rhs = op(Compose(A, C, B)) @ op(Compose(B, A, C))
        return lhs >> p >> rhs

################################################################################
# Canonical combinators

copy = canonical(lambda T: op(Copy(T)))
discard = canonical(lambda T: op(Discard(T)))

add  = canonical(lambda T: op(Add(T)))
zero = canonical(lambda T: op(Constant(T, 0)))
subtract = canonical(lambda T: op(Subtract(T)))
negate = canonical(lambda T: op(Negate(T)))

multiply = canonical(lambda T: op(Multiply(T)))

# could also call this "full".
def constant(c):
    return canonical(lambda T: op(Constant(T, c)))

# multiply by a constant
def scale(c):
    def scale_wrapper(A: FiniteFunction):
        return (constant(c)(A) @ identity(A)) >> multiply(A)
    return scale_wrapper

########################################
# comparators

gt = canonical(lambda T: op(Gt(T)))

def gt_constant(c):
    def gt_constant_wrapper(A: FiniteFunction):
        return (identity(A) @ constant(c)(A)) >> gt(A)
    return gt_constant_wrapper

################################################################################
# Other definitions

@dataclass
class Sigmoid(Lens):
    T: NdArrayType
    def source(self): return obj(self.T)
    def target(self): return obj(self.T)

    def __post_init__(self):
        if not self.T.dtype.is_floating():
            raise ValueError("Sigmoid is not defined for non-floating-point dtypes")

    def to_core(self):
        return op(defs.Sigmoid(self.T))

    def fwd(self):
        # custom fwd: run sigmoid first, and propagate its value forward, since
        # we never actually need the x value.
        A = self.source()
        return op(self) >> copy(A)

    def rev(self):
        # σ * (1 - σ) * dy
        #
        #         /----------\
        # σ(x) --●            *---\
        #         \-- (1-) --/     *---
        #                         /
        # dy   ------------------/
        T = obj(self.T)
        id_T = identity(T)
        dec_1 = (constant(1)(T) @ id_T) >> subtract(T) # 1 - x
        grad = copy(T) >> (id_T @ dec_1) >> multiply(T) # σ * (1 - σ)
        return (grad @ identity(T)) >> multiply(T) # σ * (1 - σ) * dy

sigmoid = canonical(lambda T: op(Sigmoid(T)))

def relu(X):
    return copy(X) >> (gt_constant(0)(X) @ identity(X)) >> multiply(X)

################################################################################
# Learner lenses

@dataclass
class SGD(Lens):
    T: NdArrayType
    c: ops.scalar
    def source(self): return obj(self.T)
    def target(self): return obj(self.T)
    def to_core(self): return identity(obj(self.T))
    def rev(self):
        T = obj(self.T)
        return (identity(T) @ scale(self.c)(T)) >> subtract(T)

def sgd(c: ops.scalar):
    return canonical(lambda T: op(SGD(T, c)))

@dataclass
class MSE(Lens):
    T: NdArrayType
    def source(self): return obj(self.T)
    def target(self): return obj(self.T)
    def to_core(self): return identity(obj(self.T))
    def rev(self): return subtract(obj(self.T))

mse = canonical(lambda T: op(MSE(T)))
