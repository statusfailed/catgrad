""" Linearize an OpenHypergraph into a :py:class:`FunctionDefinition`, which is
like a "flattened AST".
"""
import numpy as np
from typing import List
from dataclasses import dataclass

from open_hypergraphs import FiniteFunction, OpenHypergraph
from open_hypergraphs.numpy import layer
from catgrad.operations import operation

# An `Apply` represents the application of some operation to arguments lhs and rhs.
@dataclass
class Apply:
    """ The application of an operation of a given type to some arguments (lhs)
    producing some return values (lhs) """
    op: operation # the operation itself
    source: FiniteFunction # source type
    target: FiniteFunction # target type
    lhs: List[int] # outputs of the operation
    rhs: List[int] # args to the operation

@dataclass
class FunctionDefinition:
    """ A very pared-down version of Python's ``ast.FunctionDef`` """
    # For hypergraphs where all nodes have indegree 1,
    # :py:class:`FunctionDefinition` Static Single-Assignment (SSA) form.
    args: List[int]
    body: List[Apply]
    returns: List[int]

    @staticmethod
    def from_open_hypergraph(f: OpenHypergraph):
        # Decompose an open hypergraph.
        # First compute layering of ops, which may assign
        # multiple ops to the same layer:
        x, completed = layer(f)
        if not np.all(completed):
            raise ValueError("Cannot create a function from an OpenHypergraph with cycles")
        # now use argsort to generate a permutation, so "parallel" ops are
        # assigned different values.
        # NOTE: called twice because we need the *inverse* permutation(?)
        p: FiniteFunction = x.argsort().argsort()

        # apply permutation to the OpenHypergraph, leaving wires unchanged.
        w = FiniteFunction.identity(f.H.W)
        g: OpenHypergraph = f.permute(w, p) # permute generators by p

        # Build the FunctionDefinition
        body = [
            Apply(op=op, source=src, target=tgt, lhs=lhs.table, rhs=rhs.table)
            for op, src, tgt, rhs, lhs
            in zip(g.H.x.table, g.H.s.map_values(g.H.w), g.H.t.map_values(g.H.w), g.H.s, g.H.t)
        ]
        return FunctionDefinition(
            args=g.s.table,
            body=body,
            returns=g.t.table)
