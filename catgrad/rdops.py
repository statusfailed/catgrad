from dataclasses import dataclass
from abc import abstractmethod
from open_hypergraphs import OpenHypergraph, FiniteFunction, IndexedCoproduct, FrobeniusFunctor

from catgrad.signature import NdArrayType, obj, op, sigma_0, sigma_1
import catgrad.operations as ops
from catgrad.permutation import *

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
        raise NotImplementedError("TODO")
        # return copy >> (op(self) @ identity(A))

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
    def arrow(self): return op(ops.NAdd(self.T))
    def dagger(self): return op(NCopy(self.N, self.T))

class Constant(ops.Constant, Dagger):
    def arrow(self): return op(ops.Constant(self.T, self.x))
    def dagger(self): return op(Discard(self.T))

class Reshape(ops.Reshape, Dagger):
    def arrow(self): return op(ops.Reshape(self.T))
    def dagger(self): return op(Reshape(self.Y, self.X))

class Transpose(ops.Transpose, Dagger):
    def arrow(self): return op(ops.Transpose(self.T))
    def dagger(self): return op(Transpose(self.T))

class Multiply(ops.Multiply, Lens):
    def arrow(self): return op(ops.Multiply(self.T))
    def rev(self):
        T = self.T
        mul = self.op
        lhs = (twist(T, T) @ Copy(T).op)
        mid = (identity(T) @ twist(T, T) @ identity(T))
        rhs = mul @ mul
        return lhs >> mid >> rhs

class Compose(ops.Compose, Lens):
    def arrow(self): return op(ops.Multiply(self.T))
    def rev(self):
        A, B, C = self.A, self.B, self.C
        lhs = op(Transpose(A, B)) @ op(Transpose(B, C)) @ identity(A + C)
        p = permutation(lhs.target, [2, 1, 0, 3])
        rhs = op(Compose(A, C, B)) @ op(Compose(B, A, C))
        return lhs >> p >> rhs

################################################################################

# Translate RDOps to basic array operations
class Forget(FrobeniusFunctor):
    def map_objects(self, objects: FiniteFunction) -> IndexedCoproduct:
        return self.IndexedCoproduct().elements(objects)

    def map_operations(self, x: FiniteFunction, sources: IndexedCoproduct, targets: IndexedCoproduct) -> OpenHypergraph:
        # we lose a lot of speed here using tensor_list, but it's simpler code
        fs = [ x.arrow() for x in x.table ]
        return self.OpenHypergraph().tensor_list(fs, sigma_0, sigma_1)
