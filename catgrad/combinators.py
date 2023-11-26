""" Shorthand for useful permutations """
import numpy as np
from typing import List, Callable
from open_hypergraphs import FiniteFunction, OpenHypergraph

from catgrad.signature import sigma_0, sigma_1, NdArrayType

def unit():
    return OpenHypergraph.unit(sigma_0, sigma_1)

def identity(A: FiniteFunction):
    return OpenHypergraph.identity(A, sigma_1)

def twist(A: FiniteFunction, B: FiniteFunction):
    return OpenHypergraph.twist(A, B, sigma_1)

def transpose(A: FiniteFunction, rows: int, cols: int) -> OpenHypergraph:
    """ ``transpose`` is the permutation which transposes a (row-major) ``rows×cols`` input """
    if rows*cols != len(A):
        raise ValueError(f"rows*cols != len(A)")
    p = FiniteFunction.transpose(rows, cols)
    return OpenHypergraph.half_spider(p, A, sigma_1)

def permutation(A: FiniteFunction, p: List[int]) -> OpenHypergraph:
    if len(A) != len(p):
        raise ValueError("permutation: Object labels A must have length equal to permutation, but {len(A)} != {len(p)}")
    table = FiniteFunction.Array.array(p, FiniteFunction.Dtype)
    q = FiniteFunction(len(p), table)
    # NOTE: dagger because we want this morphism to be of type A → p(A)
    return OpenHypergraph.half_spider(q, A, sigma_1).dagger()

# Given a canonical map ``f_T : T → R₀ ● ... ● Rn`` at each generating object T,
# return a map for ``f_{T₀ ● T₁ ● ... ● Tm}``
def canonical(component: Callable[NdArrayType, OpenHypergraph]) -> Callable[FiniteFunction, OpenHypergraph]:
    """ Lift a choice of map ``f_T`` for each generating object ``T``
    into a choice for each *object* ``T₀ ● T₁ ● ... ● Tn``. """

    def canonical_wrapper(A: FiniteFunction):
        n = len(A)
        if n == 0:
            return unit() # empty diagram

        # Nonempty list of the component maps f_T for each T ∈ A.
        components = list(map(component, A))

        # Check components all have same arity/coarity
        slen = np.array([ len(f.source) for f in components ], int)
        arity = slen[0]
        if not np.all(arity == slen):
            raise ValueError("not all components have the same arity")

        tlen = np.array([ len(f.target) for f in components ], int)
        coarity = tlen[0]
        if not np.all(coarity == tlen):
            raise ValueError("not all components have the same coarity")

        # build a tensoring of all components, and wrap it in transpositions.
        f = OpenHypergraph.tensor_list(components)
        lhs = transpose(f.source, n, arity)
        rhs = transpose(f.target, n, coarity).dagger()
        return lhs >> f >> rhs

    return canonical_wrapper
