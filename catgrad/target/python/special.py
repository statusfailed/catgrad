""" special operations exclusive to the Python backend """
import ast
from typing import List, Any
from abc import abstractmethod
from open_hypergraphs import FiniteFunction, OpenHypergraph
from dataclasses import dataclass

from catgrad.signature import obj, NdArrayType

class PythonOp:
    """ Run arbitrary python code as an operation.
    Intended as an escape hatch for quick prototyping; not supported by other backends.
    """

    @abstractmethod
    def source(self) -> FiniteFunction:
        ...

    @abstractmethod
    def target(self) -> FiniteFunction:
        ...
    
    @abstractmethod
    def __call__(self, *args) -> List[Any]:
        """ return a *list* of output values """
        ...

@dataclass(frozen=True)
class IdentityPythonOp(PythonOp):
    """ The identity map, but implemented via PythonOp.
    This is not useful: it's just an example. """
    T: NdArrayType
    def source(self): return obj(self.T)
    def target(self): return obj(self.T)
    def __call__(self, x): return [x]

# Extract a list of PythonOp from an OpenHypergraph and assign each a unique symbol
def _python_op_symbol(op_id: int, op: PythonOp):
    return f"_python_op_{type(op).__name__}_{op_id}"

def _extract_python_ops(fs: dict[str, OpenHypergraph]) -> dict[str, PythonOp]:
    python_ops = [ x for f in fs.values() for x in f.H.x if isinstance(x, PythonOp) ]
    return { x: _python_op_symbol(i, x) for i, x in enumerate(python_ops) }
