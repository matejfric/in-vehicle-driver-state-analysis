import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from model.dataset import get_last_window_index
from model.memory_map import MemMapReader


@pytest.fixture(scope='module')
def dummy_memmap() -> Generator[MemMapReader, None, None]:
    """Creates a temporary memory-mapped file with dummy image data."""
    shape = (100, 8, 8)  # 100 images of 8x8
    dtype = np.uint8
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / 'test_memmap.dat'
        # Create and fill the memmap
        data = np.memmap(filepath, dtype=dtype, mode='w+', shape=shape)
        for i in range(shape[0]):
            data[i] = np.full((8, 8), i, dtype=dtype)  # fill each image with its index
        data.flush()
        reader = MemMapReader(filepath, shape=(8, 8), dtype=dtype)
        yield reader


def test_window_consecutive(dummy_memmap: MemMapReader) -> None:
    result = dummy_memmap.window(0, 4, 1)
    expected = [np.full((8, 8), i, dtype=np.uint8) for i in range(4)]
    for r, e in zip(result, expected):
        np.testing.assert_array_equal(r, e)


def test_window_with_timestep(dummy_memmap: MemMapReader) -> None:
    result = dummy_memmap.window(1, 4, 3)
    expected = [np.full((8, 8), i, dtype=np.uint8) for i in [1, 4, 7, 10]]
    for r, e in zip(result, expected):
        np.testing.assert_array_equal(r, e)


def test_get_last(dummy_memmap: MemMapReader) -> None:
    window_size = 4
    time_step = 3
    last_start = get_last_window_index(len(dummy_memmap), window_size, time_step)
    result = dummy_memmap.window(last_start, window_size, time_step)
    expected = [np.full((8, 8), i, dtype=np.uint8) for i in [90, 93, 96, 99]]
    for r, e in zip(result, expected):
        np.testing.assert_array_equal(r, e)


@settings(max_examples=200)
@given(
    size=st.integers(min_value=20, max_value=100),
    window_size=st.integers(min_value=1, max_value=10),
    time_step=st.integers(min_value=1, max_value=10),
)
def test_window_output_length(size: int, window_size: int, time_step: int) -> None:
    assume(window_size * time_step < size)
    shape = (size, 8, 8)
    dtype = np.uint8
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / 'temp.dat'
        data = np.memmap(filepath, dtype=dtype, mode='w+', shape=shape)
        for i in range(size):
            data[i] = np.full((8, 8), i, dtype=dtype)
        data.flush()
        reader = MemMapReader(filepath, shape=(8, 8), dtype=dtype)

        result = reader.window(0, window_size, time_step)
        assert len(result) == window_size


@settings(max_examples=200)
@given(
    size=st.integers(min_value=20, max_value=100),
    window_size=st.integers(min_value=1, max_value=10),
    time_step=st.integers(min_value=1, max_value=10),
)
def test_get_last_window_starts_correctly(
    size: int, window_size: int, time_step: int
) -> None:
    assume(window_size * time_step <= size)
    shape = (size, 8, 8)
    dtype = np.uint8
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / 'temp.dat'
        data = np.memmap(filepath, dtype=dtype, mode='w+', shape=shape)
        for i in range(size):
            data[i] = np.full((8, 8), i, dtype=dtype)
        data.flush()
        reader = MemMapReader(filepath, shape=(8, 8), dtype=dtype)

        try:
            last_start = get_last_window_index(len(reader), window_size, time_step)
        except ValueError:
            pytest.skip('Invalid window parameters')

        result = reader.window(last_start, window_size, time_step)
        expected_indices = list(
            range(last_start, last_start + window_size * time_step, time_step)
        )
        expected = [np.full((8, 8), i, dtype=np.uint8) for i in expected_indices]
        for r, e in zip(result, expected):
            np.testing.assert_array_equal(r, e)
