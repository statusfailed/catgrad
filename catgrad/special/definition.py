""" Definitions """

from typing import List, Set
from dataclasses import dataclass
from abc import abstractmethod
from open_hypergraphs import FiniteFunction, IndexedCoproduct, OpenHypergraph, FrobeniusFunctor
from catgrad.signature import sigma_0, sigma_1

class Definition:
    """ A ``Definition`` is an operation which can be used to extend the
    presentation of a category to include "macros".

    For example, if you want to extend the core presentation to include a
    Sigmoid operation, you would inherit from this class.
    See for example `catgrad.core.definitions.Sigmoid`.
    """

    def symbol(self):
        """ The symbol name used to identify this definition in backends which support it.
        Override this method for a custom name """
        return type(self).__name__

    # TODO: cache these? bit slow innit.
    def source(self) -> FiniteFunction:
        return self.arrow().source

    def target(self) -> FiniteFunction:
        return self.arrow().target

    @abstractmethod
    def arrow(self) -> OpenHypergraph:
        """ The morphism which this definition corresponds to """
        ...

def inline_operation(x):
    if isinstance(x, Definition):
        return x.arrow()
    return x

class Inline(FrobeniusFunctor):
    def map_objects(self, objects: FiniteFunction) -> IndexedCoproduct:
        return self.IndexedCoproduct().elements(objects)

    def map_operations(self, x: FiniteFunction, sources: IndexedCoproduct, targets: IndexedCoproduct) -> OpenHypergraph:
        fs = [ inline_operation(x) for x in x.table ]
        return self.OpenHypergraph().tensor_list(fs, sigma_0, sigma_1)

# todo: rename this "expand"?
def inline(f: OpenHypergraph) -> OpenHypergraph:
    """ inline all the definitions in a morphism by expanding them. """
    I = Inline()
    return I(f)


# A Definition is explicitly recursive when it (eventually) expands to itself.
# Note that this does not catch all types of recursion: for example, a definition like
#
#   class Recursive(Definition):
#       i: int
#       def arrow():
#           return op(Recursive(self.i + 1))
#
# ... and so we have to have a max_depth parameter to prevent unbounded recursion
def recursive_extract_definitions(defs: Set[Definition], max_depth=1024) -> dict[Definition, OpenHypergraph]:
    stack = [ (d, 1) for d in defs ]
    path = []
    result = {}

    while len(stack) > 0:
        node, depth = stack.pop()
        path = path[:depth-1] # ancestors of this node; truncate deeper paths.

        # process the node
        if depth > max_depth:
            raise ValueError("Maximum recursion depth exceeded")

        # if we've already extracted this definition, skip it.
        if node in result:
            continue

        # check for recursion
        if node in path:
            raise ValueError("Recursion detected")

        arrow = node.arrow()
        result[node] = arrow

        # recurse (add children to stack)
        path.append(node)
        children = _arrow_to_definitions(arrow)
        stack.extend( (c, depth+1) for c in children )

    return result

def _arrow_to_definitions(f: OpenHypergraph) -> Set[Definition]:
    return set( x for x in f.H.x if isinstance(x, Definition) )
