from dataclasses import dataclass
import numpy as np

from open_hypergraphs import FiniteFunction, Hypergraph, OpenHypergraph

from catgrad.signature import op, obj, sigma_0, sigma_1, NdArrayType
from catgrad.combinators import canonical

################################################################################
# Parameter-specific factorisation

@dataclass
class Parameter:
    """ ``Parameter`` is a special (non-differentiable) operation which is used
    to hide parameters when composing models.
    Before differentiating, you should factor out these operations using ``factor_parameters``.
    """
    T: NdArrayType
    def source(self): return obj()
    def target(self): return obj(self.T)

# A parameter at any object
parameter = canonical(lambda T: op(Parameter(T)))

def factor_parameters(c: OpenHypergraph):
    """ Factor ``c : A → B`` into a pair of maps
    ``d : P × A → B`` and ``p : I → P``
    such that ``c ~= (p × id) ; d``
    and ``p`` is the tensoring of all ``Parameter`` operations in ``c`` """
    # predicate determining if an operation is a parameter
    is_parameter = FiniteFunction(2, np.fromiter([int(type(x) is Parameter) for x in c.H.x], FiniteFunction.Dtype))
    return factor(c, is_parameter)

################################################################################
# Generic factorisation code

def factor(c: OpenHypergraph, p: FiniteFunction):
    """ Factor ``c : A → B`` into a pair of maps
    ``d : C × A → B`` and ``e : I → C``
    such that ``c ~= (e × id_A) ; d``
    and ``e`` consists of operations selected by predicate ``p``.
    """
    k = select(p)
    assert k.target == c.H.X
    assert len(k) <= k.target

    # find source/target wires of each of the selected operations
    e_s = c.H.s.map_indexes(k)
    e_t = c.H.t.map_indexes(k)

    # compute the tensor of selected operations, then bend around the source
    # wires so that
    #    (e₀ ● e₁ ● ... ● en) : A → B
    # becomes
    #    (e₀ ● e₁ ● ... ● en) : I → A ● B
    e = OpenHypergraph.tensor_operations(k >> c.H.x, e_s.map_values(c.H.w), e_t.map_values(c.H.w))
    e = OpenHypergraph(FiniteFunction.initial(e.H.W), e.s + e.t, e.H)

    # Remove the selected operations from c (using thattwist inverts a predicate)
    # then add the source/target wires of those operations to the left boundary.
    k = p >> FiniteFunction.twist(1,1)
    d = filter_operations(c, k)
    d = OpenHypergraph(e_s.values + e_t.values + d.s, d.t, d.H)
    return e, d

def select(p: FiniteFunction):
    """ Given a predicate X → 2, return a function
    ``f : X' → X`` such that ``f >> p == 1``.
    """
    if p.target != 2:
        raise ValueError(f"p must be a predicate, but {p.target=}")
    return FiniteFunction(len(p), p.table.nonzero()[0].astype(FiniteFunction.Dtype))

def filter_operations(f: OpenHypergraph, p: FiniteFunction):
    """ Given an OpenHypergraph ``f`` with ``X`` operations,
        and a predicate ``p : X → 2`, remove those operations ``x`` from ``d``
        for which ``p(x) == 0``.
    """
    assert f.H.X == p.source

    # k : X' → X
    k = select(p)
    H = Hypergraph(
        s = f.H.s.map_indexes(k),
        t = f.H.t.map_indexes(k),
        w = f.H.w,
        x = k >> f.H.x)
    return OpenHypergraph(f.s, f.t, H)
