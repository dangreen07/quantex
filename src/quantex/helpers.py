import numpy as np
from typing import Generic, TypeVar, cast, Any
import numpy.typing as npt

T = TypeVar("T", bound=np.generic)

class TimeNDArray(np.ndarray, Generic[T]):
    """
    Time-aware numpy array that tracks visible data length for backtesting.
    
    This class extends numpy.ndarray to provide time-series awareness for
    backtesting applications. It maintains an internal index (_i) that 
    represents how much of the array is "visible" at any given time step,
    allowing for efficient iteration over historical data without copying.
    
    Key features:
    - Maintains current time index (_i) for progressive data access
    - Supports various indexing modes (integer, slice, boolean, array)
    - Handles negative indices relative to current time index
    - Provides array conversion methods that respect time visibility
    - Supports numpy array operations while maintaining time awareness
    
    The class is designed to be used in backtesting where data becomes
    available progressively over time, and only data up to the current
    time step should be visible to the strategy.
    
    Example:
        >>> import numpy as np  
        >>> data = np.array([1, 2, 3, 4, 5])  
        >>> time_arr = TimeNDArray.from_array(data)  
        >>> print(time_arr._i)  # Shows full length  
        5  
        >>> print(time_arr.visible())  # Array up to _i  
        [1 2 3 4 5]  
        >>> time_arr._i = 3  # Only first 3 elements visible  
        >>> print(len(time_arr))  # Shows 3  
        3  
        >>> print(time_arr[:2])  # First 2 visible elements  
        [1 2]  
    """
    def __array_finalize__(self, obj: Any) -> None:
        """
        Initialize or update the time index when array is created or modified.
        
        This method is called automatically by numpy when the array is
        created or sliced. It ensures that the time index (_i) is properly
        maintained across operations.
        
        Args:
            obj (Any): The object that triggered array finalization.
        """
        if obj is None:
            return
        src_i = getattr(obj, "_i", None)
        if src_i is None:
            self._i = self.shape[0] if self.ndim > 0 else 1
        else:
            self._i = min(src_i, self.shape[0] if self.ndim > 0 else 1)

    @classmethod
    def from_array(cls, arr: npt.NDArray[T]) -> "TimeNDArray[T]":
        """
        Create a TimeNDArray from a regular numpy array.
        
        This class method provides a convenient way to convert a standard
        numpy array to a TimeNDArray with full visibility initialized.
        
        Args:
            arr (npt.NDArray[T]): Input numpy array to convert.
            
        Returns:
            TimeNDArray[T]: New TimeNDArray instance with full visibility.
            
        Example:
            >>> data = np.array([1, 2, 3])  
            >>> time_arr = TimeNDArray.from_array(data)  
            >>> print(time_arr._i)  
            3  
        """
        obj = np.asarray(arr).view(cls)
        obj._i = obj.shape[0]
        return cast(TimeNDArray[T], obj)

    def __array__(self, dtype=None): # type: ignore
        """
        Convert to numpy array, returning only visible portion.
        
        This method is called during numpy array conversions and ensures
        that only the visible portion of the data (up to _i) is returned.
        
        Args:
            dtype (optional): Target dtype for conversion.
            
        Returns:
            np.ndarray: Visible portion of the array as numpy array.
        """
        # Ensure numpy conversions see only the visible portion.
        return np.asarray(self[: self._i], dtype=dtype)

    def __repr__(self) -> str:
        """
        String representation showing only visible data.
        
        Returns:
            str: String representation of the visible portion.
        """
        return repr(np.asarray(self[: self._i]))

    def __str__(self) -> str:
        """
        String conversion showing only visible data.
        
        Returns:
            str: String representation of the visible portion.
        """
        return str(np.asarray(self[: self._i]))

    def __len__(self) -> int:
        """
        Get the current visible length of the array.
        
        Returns:
            int: Current visible length (value of _i).
        """
        return int(self._i)

    def __iter__(self): # type: ignore
        """
        Iterate over visible elements only.
        
        Yields:
            Elements from the array up to the current index.
        """
        for j in range(self._i):
            yield self[j]

    def _check_int_index(self, idx: int) -> int:
        """
        Validate and adjust integer index for time-aware access.
        
        This method handles negative indices by converting them relative
        to the current time index (_i), and ensures indices are within
        the visible bounds.
        
        Args:
            idx (int): Input index (may be negative).
            
        Returns:
            int: Adjusted index within visible bounds.
            
        Raises:
            IndexError: If index is out of visible bounds.
        """
        # Interpret negative indices relative to _i (so -1 -> _i-1).
        if idx < 0:
            idx += self._i
        if idx < 0 or idx >= self._i:
            raise IndexError("index out of bounds (beyond _i)")
        return idx

    def visible(self) -> npt.NDArray[T]:
        """
        Return a plain numpy array view of the visible portion.
        
        This method provides a clean way to get the visible portion of
        the array as a standard numpy array, suitable for operations
        that don't need time awareness.
        
        Returns:
            npt.NDArray[T]: Visible portion of the data as numpy array.
            
        Example:
            >>> time_arr._i = 3  
            >>> print(time_arr.visible())  
            [1 2 3]  
        """
        """Return a plain ndarray view of the visible portion (up to _i)."""
        return np.asarray(self[: self._i])

    def __getitem__(self, idx):
        """
        Advanced indexing with time-aware bounds checking.
        
        This method handles various indexing modes while respecting the
        current time index (_i) for bounds checking. It supports:
        - Integer indexing (with negative index conversion)
        - Slice indexing (interpreted relative to _i)
        - Boolean indexing (masked to visible portion)
        - Array indexing (with negative index conversion)
        - Tuple indexing (for multi-dimensional arrays)
        
        Args:
            idx: Index specification (int, slice, array, or tuple).
            
        Returns:
            TimeNDArray or np.ndarray: Indexed result with appropriate
                time awareness.
                
        Raises:
            IndexError: If indices are out of visible bounds.
            
        Note:
            Negative indices are converted relative to the current
            visible length (_i), not the full array length.
        """
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