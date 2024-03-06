""" Definitions """

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
        return self.arrow().source()

    def target(self) -> FiniteFunction:
        return self.arrow().target()

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
