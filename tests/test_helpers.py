import pytest
import numpy as np
from quantex.helpers import TimeNDArray


class TestTimeNDArray:
    def test_from_array(self):
        """Test creating TimeNDArray from array."""
        arr = np.array([1, 2, 3, 4, 5])
        tarr = TimeNDArray.from_array(arr)

        assert isinstance(tarr, TimeNDArray)
        assert tarr._i == 5
        assert len(tarr) == 5

    def test_len(self):
        """Test len() method."""
        arr = np.array([1, 2, 3, 4, 5])
        tarr = TimeNDArray.from_array(arr)

        assert len(tarr) == 5

        # Set _i to 3
        tarr._i = 3
        assert len(tarr) == 3

    def test_getitem_single_index(self):
        """Test single index access."""
        arr = np.array([10, 20, 30, 40, 50])
        tarr = TimeNDArray.from_array(arr)

        assert tarr[0] == 10
        assert tarr[2] == 30
        assert tarr[-1] == 50
        assert tarr[-2] == 40

    def test_getitem_slice(self):
        """Test slice access."""
        arr = np.array([10, 20, 30, 40, 50])
        tarr = TimeNDArray.from_array(arr)

        sliced = tarr[:3]
        assert isinstance(sliced, TimeNDArray)
        assert len(sliced) == 3
        assert sliced._i == 3
        np.testing.assert_array_equal(sliced, [10, 20, 30])

    def test_getitem_slice_with_i(self):
        """Test slice access with _i set."""
        arr = np.array([10, 20, 30, 40, 50])
        tarr = TimeNDArray.from_array(arr)
        tarr._i = 3

        sliced = tarr[:2]
        assert len(sliced) == 2
        np.testing.assert_array_equal(sliced, [10, 20])

    def test_getitem_negative_index(self):
        """Test negative index access."""
        arr = np.array([10, 20, 30, 40, 50])
        tarr = TimeNDArray.from_array(arr)
        tarr._i = 4

        assert tarr[-1] == 40  # _i=4, so -1 refers to index 3
        assert tarr[-2] == 30

    def test_getitem_out_of_bounds(self):
        """Test index out of bounds."""
        arr = np.array([10, 20, 30, 40, 50])
        tarr = TimeNDArray.from_array(arr)
        tarr._i = 3

        with pytest.raises(IndexError):
            _ = tarr[3]  # Index 3 is beyond _i=3

        with pytest.raises(IndexError):
            _ = tarr[-4]  # -4 goes beyond _i=3

    def test_repr_and_str(self):
        """Test string representation."""
        arr = np.array([1, 2, 3, 4, 5])
        tarr = TimeNDArray.from_array(arr)
        tarr._i = 3

        repr_str = repr(tarr)
        str_str = str(tarr)

        # Should only show visible portion
        expected = repr(np.array([1, 2, 3]))
        assert repr_str == expected
        assert str_str == str(np.array([1, 2, 3]))

    def test_iter(self):
        """Test iteration."""
        arr = np.array([10, 20, 30, 40, 50])
        tarr = TimeNDArray.from_array(arr)
        tarr._i = 3

        iterated = list(tarr)
        expected = [10, 20, 30]
        assert iterated == expected

    def test_visible(self):
        """Test visible method."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tarr = TimeNDArray.from_array(arr)
        tarr._i = 3

        visible = tarr.visible()
        assert isinstance(visible, np.ndarray)
        np.testing.assert_array_equal(visible, [1.0, 2.0, 3.0])

    def test_boolean_indexing(self):
        """Test boolean indexing."""
        arr = np.array([10, 20, 30, 40, 50])
        tarr = TimeNDArray.from_array(arr)
        tarr._i = 4

        mask = np.array([True, False, True, False, False])
        result = tarr[mask]

        assert isinstance(result, TimeNDArray)
        assert len(result) == 2
        np.testing.assert_array_equal(result, [10, 30])

    def test_array_conversion(self):
        """Test numpy array conversion."""
        arr = np.array([1, 2, 3, 4, 5])
        tarr = TimeNDArray.from_array(arr)
        tarr._i = 3

        # Test visible method
        visible = tarr.visible()
        np.testing.assert_array_equal(visible, [1, 2, 3])

        # Test __array__ method directly
        array_version = tarr.__array__()
        np.testing.assert_array_equal(array_version, [1, 2, 3])

    def test_multidimensional(self):
        """Test multidimensional TimeNDArray."""
        arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        tarr = TimeNDArray.from_array(arr)
        tarr._i = 3

        # Test slicing first dimension
        sliced = tarr[:2]
        assert sliced.shape == (2, 2)
        assert isinstance(sliced, TimeNDArray)
        assert sliced._i == 2

        # Test accessing second dimension
        col = tarr[:, 1]
        assert isinstance(col, TimeNDArray)
        np.testing.assert_array_equal(col, [2, 4, 6])