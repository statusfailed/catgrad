""" Useful definitions which extend :py:mod:`catgrad.core.operation` with
additional ops that can be "inlined" or expanded in terms of only core
operations.

For example, the ``Sigmoid`` pseudo-operation expands to  ``1 / (1 + exp(-x))``.
"""
from dataclasses import dataclass

from catgrad.signature import obj, op, NdArrayType, Dtype
import catgrad.core.operation as ops
from catgrad.combinators import identity
from catgrad.special.definition import Definition

@dataclass(frozen=True)
class Sigmoid(Definition):
    """ The sigmoid function as a Definition """
    T: NdArrayType

    def __post_init__(self):
        if not self.T.dtype.is_floating():
            raise ValueError("Sigmoid is not defined for non-floating-point dtypes")

    # override source/target for speeeed
    def source(self):
        return obj(self.T)

    def target(self):
        return obj(self.T)

    def arrow(self):
        T = self.T
        U = NdArrayType((), T.dtype)

        full1 = op(ops.Constant(T, 1))

        # inc(x) := 1 + x
        inc = (full1 @ identity(obj(T))) >> op(ops.Add(T))

        # den(x) := 1 + exp(-x)
        den = op(ops.Negate(T)) >> ops.exp1(T) >> inc

        #      1
        # -----------
        # 1 + exp(-x)
        return (full1 @ den) >> op(ops.Divide(T))
