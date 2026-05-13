"""Area weighting utilities for EOF analysis."""
import numpy as np


def area_weights(lats: np.ndarray) -> np.ndarray:
    """
    Return sqrt(cos(lat)) weights for a 1-D latitude array.

    Used before EOF decomposition so that high-latitude grid cells
    (which cover less area) do not dominate the variance.
    """
    return np.sqrt(np.cos(np.deg2rad(lats)).clip(min=0))


def area_weights_2d(lats_2d: np.ndarray) -> np.ndarray:
    """Return sqrt(cos(lat)) weights broadcast to a 2-D (lat, lon) grid."""
    return area_weights(lats_2d[:, 0])[:, np.newaxis] * np.ones_like(lats_2d)
