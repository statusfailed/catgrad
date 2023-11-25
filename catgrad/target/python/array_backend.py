from typing import Protocol, TypeVar, Type, Any, List, Tuple
import numpy as np

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
    :py:mod:`catgrad.operations`.
    See for example the :py:class:`Numpy` backend. """
    @staticmethod
    def dtype(d: signature.Dtype) -> Any:
        ...

    @staticmethod
    def constant(x: Any) -> A:
        ...

    @staticmethod
    def ncopy(shape: Tuple, x: A) -> A:
        ...

    @staticmethod
    def nadd(shape: Tuple, x: A) -> A:
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

# A Numpy array backend
class Numpy:
    @staticmethod
    def dtype(d: signature.Dtype) -> Any:
        match d:
            case signature.Dtype.int32: return np.int32
            case signature.Dtype.float32: return np.float32
            case x: raise ValueError(f"dtype {x} is not implemented for Numpy")

    @staticmethod
    def constant(x: Any, dtype: signature.Dtype) -> np.ndarray:
        return np.array(x, dtype=Numpy.dtype(dtype))

    @staticmethod
    def ncopy(shape: Tuple, x: np.ndarray) -> np.ndarray:
        return np.broadcast_to(x, (*shape, *x.shape))

    @staticmethod
    def nadd(shape: Tuple, x: np.ndarray) -> np.ndarray:
        return x.sum(tuple(range(len(shape))))

    @staticmethod
    def reshape(x: np.ndarray, shape: Tuple) -> np.ndarray:
        return np.reshape(x, shape)

    @staticmethod
    def permute(x: np.ndarray, p: List[int]):
        return np.transpose(x, p)

    @staticmethod
    def compose(x: np.ndarray, y: np.ndarray, axes: int) -> np.ndarray:
        return np.tensordot(x, y, axes=axes)
