""" Generate expressions for :py:class:`catgrad.special.definition.Definition` operations.
Note that this is an *extension* to the core operations: you can remove
definitions by inlining them with
:py:func:`catgrad.special.definition.inline``.
"""
import ast
from typing import Tuple, List
from collections import Counter

from open_hypergraphs import OpenHypergraph

from catgrad.special.definition import Definition, recursive_extract_definitions
from catgrad.target.ast import Apply
from catgrad.target.python.codegen.operation import load, store

def _extract_definitions(fs: dict[str, OpenHypergraph]) -> dict[Definition, Tuple[str, OpenHypergraph]]:
    """ For each unique Definition op in a core+defs OpenHypergraph, extract a
    pair of (Symbol, OpenHypergraph) representing the *name* of that definition
    and its expansion.

    Note that expansions may refer to one another, but must not be recursive.
    You can slip recursion through, but don't.
    """

    # get the set of unique Definition ops in all morphisms supplied
    def_ops = set( x for f in fs.values() for x in f.H.x if isinstance(x, Definition) )
    def_ops = recursive_extract_definitions(def_ops)

    symbol_count = Counter()
    result = {}
    for x in def_ops:
        # definitions can have the same symbol, we disambiguate by appending an integer.
        symbol = x.symbol()
        symbol_id = symbol_count[symbol]
        symbol_count[symbol] += 1

        # no need to worry about overwriting, each x is unique - from a set
        result[x] = (f"_def_{symbol}_{symbol_id}", x.arrow())

    return result

def definition(apply: Apply, symbol: str) -> List[ast.Assign]:
    # unlike expr_lhs, the LHS is *always* a tuple: functions always return a *list* of results.
    lhs = ast.Tuple(elts=[ store(i) for i in apply.lhs ], ctx=ast.Store())
    args = [ load(i) for i in apply.rhs ]
    func = ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr=symbol, ctx=ast.Load())
    expr = ast.Call(func=func, args=args, keywords=[])
    assignment = ast.Assign(targets=[lhs], value=expr)
    return [assignment]
