from open_hypergraphs import FiniteFunction, IndexedCoproduct, OpenHypergraph
from open_hypergraphs import Functor, FrobeniusFunctor, Optic

from catgrad.signature import sigma_0, sigma_1

# Translate RDOps to basic array operations
class Forget(FrobeniusFunctor):
    def map_objects(self, objects: FiniteFunction) -> IndexedCoproduct:
        return self.IndexedCoproduct().elements(objects)

    def map_operations(self, x: FiniteFunction, sources: IndexedCoproduct, targets: IndexedCoproduct) -> OpenHypergraph:
        # we lose a lot of speed here using tensor_list, but it's simpler code
        fs = [ x.arrow() for x in x.table ]
        return self.OpenHypergraph().tensor_list(fs, sigma_0, sigma_1)

# Map rdops to their fwd maps
class Fwd(FrobeniusFunctor):
    def map_objects(self, objects: FiniteFunction) -> IndexedCoproduct:
        return self.IndexedCoproduct().elements(objects)

    def map_operations(self, x: FiniteFunction, sources: IndexedCoproduct, targets: IndexedCoproduct) -> OpenHypergraph:
        fs = [ x.fwd() for x in x.table ]
        return self.OpenHypergraph().tensor_list(fs, sigma_0, sigma_1)

# Map rdops to their reverse maps
class Rev(FrobeniusFunctor):
    def map_objects(self, objects: FiniteFunction) -> IndexedCoproduct:
        return self.IndexedCoproduct().elements(objects)

    def map_operations(self, x: FiniteFunction, sources: IndexedCoproduct, targets: IndexedCoproduct) -> OpenHypergraph:
        fs = [ x.rev() for x in x.table ]
        return self.OpenHypergraph().tensor_list(fs, sigma_0, sigma_1)

# Map arrows into their 'bidirectional' form consisting of both fwd and rev passes in opposite directions
class Bidirectional(Optic):
    F = Fwd()
    R = Rev()

    def residual(self, x: FiniteFunction, A: IndexedCoproduct, B: IndexedCoproduct) -> IndexedCoproduct:
        return IndexedCoproduct.from_list(None, [op.residual() for op in x], dtype=object)
