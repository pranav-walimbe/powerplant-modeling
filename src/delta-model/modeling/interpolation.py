"""Raster interpolation utilities for missing NO2 pixels."""

import numpy as np
import torch

from config import MODEL_IMAGE_CHANNELS


def bilinear_interpolation_fill(image: torch.Tensor) -> torch.Tensor:
    """Fill missing NO2 by separable linear interpolation on the raster grid.

    Args:
        image: CPU tensor containing normalized image channels and visible mask.

    Returns:
        Complete normalized NO2 raster.
    """
    no2 = image[:, 0].numpy()
    visible = image[:, MODEL_IMAGE_CHANNELS].numpy().astype(bool)
    predictions = no2.copy()
    rows = np.arange(no2.shape[1])
    columns = np.arange(no2.shape[2])

    for sample in range(no2.shape[0]):
        fallback = float(no2[sample][visible[sample]].mean())
        horizontal = np.empty_like(no2[sample])
        vertical = np.empty_like(no2[sample])
        for row in rows:
            support = visible[sample, row]
            horizontal[row] = (
                np.interp(columns, columns[support], no2[sample, row, support]) if support.any() else fallback
            )
        for column in columns:
            support = visible[sample, :, column]
            vertical[:, column] = (
                np.interp(rows, rows[support], no2[sample, support, column]) if support.any() else fallback
            )
        missing = ~visible[sample]
        predictions[sample, missing] = 0.5 * (horizontal[missing] + vertical[missing])
    return torch.from_numpy(predictions[:, None])
