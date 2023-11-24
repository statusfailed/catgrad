""" Shorthand for useful permutations """
from typing import List
from open_hypergraphs import FiniteFunction, OpenHypergraph

from catgrad.signature import sigma_0, sigma_1

def unit():
    return OpenHypergraph.unit(sigma_0, sigma_1)

def identity(A: FiniteFunction):
    return OpenHypergraph.identity(A, sigma_1)

def twist(A: FiniteFunction, B: FiniteFunction):
    return OpenHypergraph.twist(A, B, sigma_1)

def transpose(A: FiniteFunction, B: FiniteFunction) -> OpenHypergraph:
    s = FiniteFunction.identity(len(A) + len(B))
    t = FiniteFunction.transpose(len(A), len(B))
    return OpenHypergraph.spider(s, t, sigma_0, sigma_1)

def permutation(A: FiniteFunction, p: List[int]) -> OpenHypergraph:
    if len(A) != len(p):
        raise ValueError("permutation: Object labels A must have length equal to permutation, but {len(A)} != {len(p)}")
    table = FiniteFunction.Array.array(p, FiniteFunction.Dtype)
    q = FiniteFunction(len(p), table)
    # NOTE: dagger because we want this morphism to be of type A → p(A)
    return OpenHypergraph.half_spider(q, A, sigma_1).dagger()
