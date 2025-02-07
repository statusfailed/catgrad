from typing import Any, List, Tuple
import torch
from catgrad import signature

# A Torch array backend
class Torch:
    @staticmethod
    def dtype(d: signature.Dtype) -> Any:
        match d:
            case signature.Dtype.int32: return torch.int32
            case signature.Dtype.float32: return torch.float32
            case signature.Dtype.bool: return torch.bool
            case x: raise ValueError(f"dtype {x} is not implemented for Torch")

    @staticmethod
    def constant(x: Any, shape: Tuple, dtype: signature.Dtype) -> torch.Tensor:
        return torch.tensor(x, dtype=Torch.dtype(dtype))

    @staticmethod
    def ncopy(shape: Tuple, x: torch.Tensor) -> torch.Tensor:
        return torch.broadcast_to(x.reshape(x.shape + (1,)*len(shape)), x.shape + shape)

    @staticmethod
    def nsplit(x: torch.Tensor, k: int) -> List[torch.Tensor]:
        if k == 0:
            return None

        # NOTE: this interface is subtly different to numpy!
        # You specify the *size* of the chunk, not the number of splits!
        result = [ a.squeeze(-1) for a in torch.split(x, 1, dim=-1) ]
        if k == 1:
            return result[0] # unpack list for single values
        return result

    @staticmethod
    def nconcatenate(xs: List[torch.Tensor], k: int) -> List[torch.Tensor]:
        assert len(xs) == k
        if k == 0: return None
        xs = [ a.unsqueeze(-1) for a in xs ]
        return torch.concatenate(xs, dim=-1)

    @staticmethod
    def nadd(dims: Tuple, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dims)

    @staticmethod
    def nmax(dims: Tuple, x: torch.Tensor) -> torch.Tensor:
        # TODO: FIXME: this doesn't work if dims aren't (-N, -(N+1), ...)
        return x.flatten(start_dim=min(dims)).max(dim=-1).values

    @staticmethod
    def reshape(x: torch.Tensor, shape: Tuple) -> torch.Tensor:
        return torch.reshape(x, shape)

    @staticmethod
    def permute(x: torch.Tensor, p: List[int]):
        return torch.permute(x, p)

    @staticmethod
    def compose(x: torch.Tensor, y: torch.Tensor, axes: int) -> torch.Tensor:
        return torch.tensordot(x, y, dims=axes)
