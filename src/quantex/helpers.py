import numpy as np
from typing import Generic, TypeVar, cast, Any
import numpy.typing as npt

T = TypeVar("T", bound=np.generic)

class TimeNDArray(np.ndarray, Generic[T]):
    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        src_i = getattr(obj, "_i", None)
        if src_i is None:
            self._i = self.shape[0] if self.ndim > 0 else 1
        else:
            self._i = min(src_i, self.shape[0] if self.ndim > 0 else 1)

    @classmethod
    def from_array(cls, arr: npt.NDArray[T]) -> "TimeNDArray[T]":
        obj = np.asarray(arr).view(cls)
        obj._i = obj.shape[0]
        return cast(TimeNDArray[T], obj)

    def __array__(self, dtype=None): # type: ignore
        # Ensure numpy conversions see only the visible portion.
        return np.asarray(self[: self._i], dtype=dtype)

    def __repr__(self) -> str:
        return repr(np.asarray(self[: self._i]))

    def __str__(self) -> str:
        return str(np.asarray(self[: self._i]))

    def __len__(self) -> int:
        return int(self._i)

    def __iter__(self): # type: ignore
        for j in range(self._i):
            yield self[j]

    def _check_int_index(self, idx: int) -> int:
        # Interpret negative indices relative to _i (so -1 -> _i-1).
        if idx < 0:
            idx += self._i
        if idx < 0 or idx >= self._i:
            raise IndexError("index out of bounds (beyond _i)")
        return idx

    def __getitem__(self, idx):
        import numpy as _np

        # Ellipsis / omitted -> visible slice
        if idx is None or idx is Ellipsis:
            return super().__getitem__(slice(0, self._i))

        # Tuple indexing: treat first axis specially
        if isinstance(idx, tuple):
            first, rest = idx[0], idx[1:]

            # integer first-axis
            if isinstance(first, int):
                first = self._check_int_index(first)
                new_idx = (first,) + rest
                return super().__getitem__(new_idx)

            # slice on first axis: interpret relative to _i
            if isinstance(first, slice):
                start, stop, step = first.indices(self._i)
                new_first = slice(start, stop, step)
                new_idx = (new_first,) + rest
                res = super().__getitem__(new_idx)
                if isinstance(res, TimeNDArray):
                    res._i = res.shape[0] if res.ndim > 0 else 1
                return res

            # numpy array for first axis
            if isinstance(first, _np.ndarray):
                if first.dtype == _np.bool_:
                    if first.shape[0] != self.shape[0]:
                        raise IndexError(
                            "boolean index must be same length as axis 0"
                        )
                    mask = first.copy()
                    mask[self._i :] = False
                    new_idx = (mask,) + rest
                    res = super().__getitem__(new_idx)
                    if isinstance(res, TimeNDArray):
                        res._i = res.shape[0] if res.ndim > 0 else 1
                    return res
                else:
                    arr = first.copy()
                    # negative entries reference relative to _i
                    arr[arr < 0] += self._i
                    if (arr < 0).any() or (arr >= self._i).any():
                        raise IndexError("index out of bounds (beyond _i)")
                    new_idx = (arr,) + rest
                    res = super().__getitem__(new_idx)
                    if isinstance(res, TimeNDArray):
                        res._i = res.shape[0] if res.ndim > 0 else 1
                    return res

        # Single integer index
        if isinstance(idx, int):
            idx = self._check_int_index(idx)
            return super().__getitem__(idx)

        # Single slice -> interpret relative to _i
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._i)
            return super().__getitem__(slice(start, stop, step))

        # numpy array as single index
        if isinstance(idx, _np.ndarray):
            if idx.dtype == _np.bool_:
                if idx.shape[0] != self.shape[0]:
                    raise IndexError(
                        "boolean index must be same length as axis 0"
                    )
                mask = idx.copy()
                mask[self._i :] = False
                res = super().__getitem__(mask)
                if isinstance(res, TimeNDArray):
                    res._i = res.shape[0] if res.ndim > 0 else 1
                return res
            else:
                arr = idx.copy()
                arr[arr < 0] += self._i
                if (arr < 0).any() or (arr >= self._i).any():
                    raise IndexError("index out of bounds (beyond _i)")
                res = super().__getitem__(arr)
                if isinstance(res, TimeNDArray):
                    res._i = res.shape[0] if res.ndim > 0 else 1
                return res

        # Fallback: do the indexing, then convert/truncate axis 0 visibility.
        res = super().__getitem__(idx)
        if isinstance(res, TimeNDArray) and res.ndim >= 1:
            res._i = min(res._i, res.shape[0])
        elif isinstance(res, np.ndarray) and res.ndim >= 1:
            ta = res.view(TimeNDArray)
            ta._i = min(self._i, ta.shape[0] if ta.ndim > 0 else 1)
            return ta
        return res

    def visible(self) -> npt.NDArray[T]:
        """Return a plain ndarray view of the visible portion (up to _i)."""
        return np.asarray(self[: self._i])