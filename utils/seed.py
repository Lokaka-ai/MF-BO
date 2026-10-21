import os
import random
from typing import Any


def set_global_seed(seed: int) -> None:
    """Set random seeds across common libraries for reproducibility."""

    random.seed(seed)

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
        
    except Exception:
        pass


def capture_random_state() -> tuple[list[Any], dict[str, Any] | None]:
    """Return JSON-serializable Python and NumPy RNG states."""
    python_state = _tuple_to_list(random.getstate())
    numpy_state = None
    try:
        import numpy as np  # type: ignore

        state = np.random.get_state()
        numpy_state = {
            "bit_generator": state[0],
            "keys": state[1].tolist(),
            "position": state[2],
            "has_gauss": state[3],
            "cached_gaussian": state[4],
        }
    except Exception:
        pass
    return python_state, numpy_state


def restore_random_state(
    python_state: list[Any] | None,
    numpy_state: dict[str, Any] | None,
) -> None:
    """Restore RNG states saved by :func:`capture_random_state`."""
    if python_state is not None:
        random.setstate(_list_to_tuple(python_state))

    if numpy_state is not None:
        try:
            import numpy as np  # type: ignore

            np.random.set_state((
                numpy_state["bit_generator"],
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            ))
        except Exception:
            pass


def _tuple_to_list(value):
    if isinstance(value, tuple):
        return [_tuple_to_list(item) for item in value]
    return value


def _list_to_tuple(value):
    if isinstance(value, list):
        return tuple(_list_to_tuple(item) for item in value)
    return value

