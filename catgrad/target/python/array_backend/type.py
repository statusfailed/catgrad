from typing import Protocol, TypeVar, Type, Any, List, Tuple
from catgrad import signature

# Generic NdArray type, supporting pointwise arithmetic and matmul.
class NdArray(Protocol):
    shape: Tuple
    dtype: Any

    def __add__(self, other):
        ...

    def __sub__(self, other):
        ...

    def __mul__(self, other):
        ...

    def __div__(self, other):
        ...

    def __matmul__(self, other):
        ...

A = TypeVar('A', bound=NdArray)

class ArrayBackend(Protocol[A]):
    """ An ArrayBackend is an implementation of the basic array operations in
    :py:mod:`catgrad.core.operation`.
    See for example the :py:class:`Numpy` backend. """
    @staticmethod
    def dtype(d: signature.Dtype) -> Any:
        ...

    @staticmethod
    def constant(x: Any, shape: Tuple, dtype: signature.Dtype) -> A:
        ...

    @staticmethod
    def ncopy(shape: Tuple, x: A) -> A:
        ...

    @staticmethod
    def nsplit(x: A, k: int) -> List[A]:
        ...

    @staticmethod
    def nconcatenate(xs: List[A], k: int) -> List[A]:
        ...

    @staticmethod
    def nadd(dims: Tuple, x: A) -> A:
        ...

    @staticmethod
    def nmax(dims: Tuple, x: A) -> A:
        ...

    @staticmethod
    def reshape(x: A, shape: Tuple) -> A:
        ...

    @staticmethod
    def permute(x: A, p: List[int]):
        ...

    @staticmethod
    def compose(x: A, y: A, axes: int) -> A:
        ...
