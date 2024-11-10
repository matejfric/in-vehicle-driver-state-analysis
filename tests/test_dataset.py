import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from model.dataset import get_last_window_index


# This mimics the behavior of the `window` method in the `MemMapReader` class.
def window(
    x: np.ndarray, start: int, window_size: int, time_step: int = 1
) -> list[np.ndarray]:
    """Return a list of `window_size` elements taking every `time_step`-th element. For `time_step=1` returns consecutive elements."""
    end = min(start + window_size * time_step, len(x))
    return [x[i] for i in range(start, end, time_step)]


def test_window_consecutive() -> None:
    x = np.arange(10)
    result = window(x, 0, 4, 1)
    expected = [0, 1, 2, 3]
    assert result == expected


def test_window_with_timestep() -> None:
    x = np.arange(11)
    result = window(x, 1, 4, 3)
    expected = [1, 4, 7, 10]
    assert result == expected


def test_get_last() -> None:
    window_size = 4
    time_step = 3
    x = np.arange(100)
    last_start = get_last_window_index(len(x), window_size, time_step)
    result = window(x, last_start, window_size, time_step)
    expected = [90, 93, 96, 99]  # Expected last valid window
    assert result == expected


@settings(max_examples=200)
@given(
    x=st.lists(st.integers(), min_size=10, max_size=1_000),
    window_size=st.integers(min_value=1, max_value=10),
    time_step=st.integers(min_value=1, max_value=10),
)
def test_window_output_length(x: list[int], window_size: int, time_step: int) -> None:
    x = np.array(x)

    # This would return a list smaller than the window size, not a valid window.
    assume(window_size * time_step < len(x))

    start = 0
    result = window(x, start, window_size, time_step)
    assert len(result) == window_size


@settings(max_examples=200)
@given(
    x=st.lists(st.integers(), min_size=1, max_size=1_000),
    window_size=st.integers(min_value=1, max_value=10),
    time_step=st.integers(min_value=1, max_value=10),
)
def test_get_last_window_starts_correctly(
    x: list[int], window_size: int, time_step: int
) -> None:
    x = np.array(x)

    try:
        last_start = get_last_window_index(len(x), window_size, time_step)
    except ValueError:
        assert window_size * time_step > len(x), 'Raised ValueError for valid input.'
        pytest.skip('Invalid window size.')

    result = window(x, last_start, window_size, time_step)
    assert len(result) == window_size
    assert result == [x[i] for i in range(last_start, len(x), time_step)]
