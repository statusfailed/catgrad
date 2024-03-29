from typing import Any, List, Tuple
import torch
from catgrad import signature

# A Numpy array backend
class Torch:
    @staticmethod
    def dtype(d: signature.Dtype) -> Any:
        match d:
            case signature.Dtype.int32: return torch.int32
            case signature.Dtype.float32: return torch.float32
            case signature.Dtype.float32: return torch.bool
            case x: raise ValueError(f"dtype {x} is not implemented for Torch")

    @staticmethod
    def constant(x: Any, shape: Tuple, dtype: signature.Dtype) -> torch.tensor:
        return torch.tensor(x, dtype=Torch.dtype(dtype))

    @staticmethod
    def ncopy(shape: Tuple, x: torch.tensor) -> torch.tensor:
        return torch.broadcast_to(x.reshape(x.shape + (1,)*len(shape)), x.shape + shape)

    @staticmethod
    def nsplit(x: torch.tensor, k: int) -> List[torch.tensor]:
        if k == 0:
            return None

        # NOTE: this interface is subtly different to numpy!
        # You specify the *size* of the chunk, not the number of splits!
        result = torch.split(x, 1, dim=-1)
        if k == 1:
            return result[0] # unpack list for single values
        return result

    @staticmethod
    def nconcatenate(xs: List[torch.tensor], k: int) -> List[torch.tensor]:
        assert len(xs) == k
        if k == 0: return None
        return torch.concatenate(xs, axis=-1)

    @staticmethod
    def nadd(dims: Tuple, x: torch.tensor) -> torch.tensor:
        return x.sum(dims)

    @staticmethod
    def nmax(dims: Tuple, x: torch.tensor) -> torch.tensor:
        # TODO: FIXME: this doesn't work if dims aren't (-N, -(N+1), ...)
        return x.flatten(start_dim=min(dims)).max(dim=-1).values

    @staticmethod
    def reshape(x: torch.tensor, shape: Tuple) -> torch.tensor:
        return torch.reshape(x, shape)

    @staticmethod
    def permute(x: torch.tensor, p: List[int]):
        return torch.permute(x, p)

    @staticmethod
    def compose(x: torch.tensor, y: torch.tensor, axes: int) -> torch.tensor:
        return torch.tensordot(x, y, dims=axes)
