from typing import Any, List, Tuple
import numpy as np
from catgrad import signature

# A Numpy array backend
class Numpy:
    @staticmethod
    def dtype(d: signature.Dtype) -> Any:
        match d:
            case signature.Dtype.int32: return np.int32
            case signature.Dtype.float32: return np.float32
            case signature.Dtype.bool: return bool # using np.bool gets a warning
            case x: raise ValueError(f"dtype {x} is not implemented for Numpy")

    @staticmethod
    def constant(x: Any, shape: Tuple, dtype: signature.Dtype) -> np.ndarray:
        # TODO: does this allocate the full array? Check perf
        return np.full(shape, x, dtype=Numpy.dtype(dtype))

    @staticmethod
    def ncopy(shape: Tuple, x: np.ndarray) -> np.ndarray:
        return np.broadcast_to(x.reshape(x.shape + (1,)*len(shape)), x.shape + shape)

    @staticmethod
    def nsplit(x: np.ndarray, k: int) -> List[np.ndarray]:
        if k == 0:
            return None

        result = np.split(x, k, -1)
        if k == 1:
            return result[0] # unpack list for single values
        return result

    @staticmethod
    def nconcatenate(xs: List[np.ndarray], k: int) -> List[np.ndarray]:
        assert len(xs) == k
        if k == 0: return None
        return np.concatenate(xs, axis=-1)

    @staticmethod
    def nadd(dims: Tuple, x: np.ndarray) -> np.ndarray:
        return x.sum(dims)

    @staticmethod
    def nmax(dims: Tuple, x: np.ndarray) -> np.ndarray:
        return x.max(dims)

    @staticmethod
    def reshape(x: np.ndarray, shape: Tuple) -> np.ndarray:
        return np.reshape(x, shape)

    @staticmethod
    def permute(x: np.ndarray, p: List[int]):
        return np.transpose(x, p)

    @staticmethod
    def compose(x: np.ndarray, y: np.ndarray, axes: int) -> np.ndarray:
        return np.tensordot(x, y, axes=axes)
