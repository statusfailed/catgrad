import numpy as np
from typing import List

from open_hypergraphs import OpenHypergraph, Hypergraph

def assert_equal(xs: List[np.ndarray], ys: List[np.ndarray]):
    assert len(xs) == len(ys)
    for x, y in zip(xs, ys):
        assert x.shape == y.shape
        assert x.dtype == y.dtype
        assert np.array_equal(x, y, equal_nan=True)

# NOTE: this method lifted from OpenHypergraphs test suite!
# TODO: replace with real equality checking when available
def assert_open_hypergraphs_equal(f: OpenHypergraph, g: OpenHypergraph):
    """ Not-quite-equality checking for OpenHypergraph. """
    # same type
    assert f.source == g.source
    assert f.target == g.target

    assert_hypergraphs_equal(f.H, g.H)

# NOTE: this method lifted from OpenHypergraphs test suite!
# TODO: replace with real equality checking when available
def assert_hypergraphs_equal(H: Hypergraph, G: Hypergraph):
    """ Not-quite-equality checking for Hypergraphs. """

    # same number of wires, wire labels
    assert H.W == G.W
    # i = H.w.argsort()
    # j = G.w.argsort()
    # assert (i >> H.w) == (j >> G.w)

    # same number of edges, edge labels
    assert H.X == G.X
    # i = H.x.argsort()
    # j = G.x.argsort()
    # assert (i >> H.x) == (j >> G.x)
