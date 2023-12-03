import numpy as np
from dataclasses import dataclass
from abc import abstractmethod
from open_hypergraphs import OpenHypergraph, FiniteFunction, IndexedCoproduct, FrobeniusFunctor

from catgrad.signature import NdArrayType, obj, op, sigma_0, sigma_1
import catgrad.operations as ops
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
    def arrow(self) -> OpenHypergraph:
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
    def arrow(self): return op(ops.Copy(self.T))
    def dagger(self): return op(Add(self.T))

class NCopy(ops.NCopy, Dagger):
    def arrow(self): return op(ops.NCopy(self.N, self.T))
    def dagger(self): return op(NAdd(self.N, self.T))

class Discard(ops.Discard, Dagger):
    def arrow(self): return op(ops.Discard(self.T))
    # NOTE: we do one scalar zero and broadcast into the correct shape
    def dagger(self):
        U = NdArrayType((), self.T.dtype)
        return op(Constant(U, 0)) >> op(NCopy(self.T, U))

class Add(ops.Add, Dagger):
    def arrow(self): return op(ops.Add(self.T))
    def dagger(self): return op(Copy(self.T))

class NAdd(ops.NAdd, Dagger):
    def arrow(self): return op(ops.NAdd(self.N, self.T))
    def dagger(self): return op(NCopy(self.N, self.T))

class Subtract(ops.Add, Dagger):
    def arrow(self): return op(ops.Subtract(self.T))
    def dagger(self):
        T = obj(self.T)
        return copy(T) >> (identity(T) @ negate(T))

class Negate(ops.Negate, Dagger):
    def arrow(self): return op(ops.Negate(self.T))
    def dagger(self): return op(self)

class Constant(ops.Constant, Dagger):
    def arrow(self): return op(ops.Constant(self.T, self.x))
    def dagger(self): return op(Discard(self.T))

class Reshape(ops.Reshape, Dagger):
    def arrow(self): return op(ops.Reshape(self.X, self.Y))
    def dagger(self): return op(Reshape(self.Y, self.X))

class Permute(ops.Permute, Dagger):
    def arrow(self): return op(ops.Permute(self.T, self.p))
    def dagger(self): return op(Permute(self.T, np.argsort(self.p).tolist()))

class Multiply(ops.Multiply, Lens):
    def arrow(self): return op(ops.Multiply(self.T))
    def rev(self):
        T = obj(self.T)
        mul = op(self)
        lhs = (twist(T, T) @ copy(T))
        mid = (identity(T) @ twist(T, T) @ identity(T))
        rhs = mul @ mul
        return lhs >> mid >> rhs

class Compose(ops.Compose, Lens):
    def arrow(self): return op(ops.Compose(self.A, self.B, self.C))
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

def full1(c: ops.scalar):
    """ ``full1(c)`` returns a function f(T: NdArrayType) which constructs a
    circuit of type ``I → T`` representing the constant array of type T filled
    with values ``c``. """
    def full1_wrapped(T: NdArrayType):
        U = NdArrayType((), T.dtype)
        a = op(Constant(U, c))
        b = op(NCopy(T, U))
        return a >> b
    return full1_wrapped

copy = canonical(lambda T: op(Copy(T)))
discard = canonical(lambda T: op(Discard(T)))

add  = canonical(lambda T: op(Add(T)))
zero = canonical(full1(0))
subtract = canonical(lambda T: op(Subtract(T)))
negate = canonical(lambda T: op(Negate(T)))

multiply = canonical(lambda T: op(Multiply(T)))
# could also call this "full".
constant = lambda c: canonical(full1(c))

# multiply by a constant
def scale(c):
    def scale_wrapper(A: FiniteFunction):
        return (constant(c)(A) @ identity(A)) >> multiply(A)
    return scale_wrapper

################################################################################

# Translate RDOps to basic array operations
class Forget(FrobeniusFunctor):
    def map_objects(self, objects: FiniteFunction) -> IndexedCoproduct:
        return self.IndexedCoproduct().elements(objects)

    def map_operations(self, x: FiniteFunction, sources: IndexedCoproduct, targets: IndexedCoproduct) -> OpenHypergraph:
        # we lose a lot of speed here using tensor_list, but it's simpler code
        fs = [ x.arrow() for x in x.table ]
        return self.OpenHypergraph().tensor_list(fs, sigma_0, sigma_1)

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

    # TODO: tidy me
    def arrow(self):
        # x → exp(x) / (1 + exp(x))
        # x → 1 / (1 + exp(-x))
        T = self.T
        U = NdArrayType((), T.dtype)

        # the 1 constant at shape T
        full1 = op(ops.Constant(U, 1)) >> op(ops.NCopy(T, U))
        inc = (full1 @ identity(obj(T))) >> op(ops.Add(T))

        den = op(ops.Negate(T)) >> ops.exp1(T) >> inc
        return (full1 @ den) >> op(ops.Divide(T))

    def rev(self):
        # TODO: implement arithmetic operations on arrows so we can write
        # σ * (1 - σ) * dy
        T = obj(self.T)
        σ = op(self)
        f = (constant(1)(T) @ σ) >> subtract(T) # 1 - σ
        grad = copy(T) >> (σ @ f) >> multiply(T) # σ * (1 - σ)
        return (grad @ identity(T)) >> multiply(T) # σ * (1 - σ) * dy

sigmoid = canonical(lambda T: op(Sigmoid(T)))

################################################################################
# Learner lenses

@dataclass
class SGD(Lens):
    T: NdArrayType
    c: ops.scalar
    def source(self): return obj(self.T)
    def target(self): return obj(self.T)
    def arrow(self): return identity(obj(self.T))
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
    def arrow(self): return identity(obj(self.T))
    def rev(self): return subtract(obj(self.T))

mse = canonical(lambda T: op(MSE(T)))
