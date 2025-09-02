# axolotl/preprocessing.py
"""
Functions for preprocessing raw electrophysiology data, such as baseline correction.
"""

import numpy as np
from scipy.stats import trim_mean

def compute_baselines_int16_deriv_robust(raw_data, segment_len=100_000, diff_thresh=50, trim_fraction=0.05):
    """
    Compute mean baseline per channel over non-overlapping segments,
    using derivative masking + trimmed mean to suppress spike influence.

    Parameters
    ----------
    raw_data : np.ndarray, shape (T, C)
        The raw data in int16 format.
    segment_len : int
        The length of each segment in samples for baseline calculation.
    diff_thresh : int
        The derivative threshold to mask out spike-like events.
    trim_fraction : float
        The fraction of data to trim from each end before computing the mean.

    Returns
    -------
    np.ndarray, shape (C, n_segments)
        An array of baseline values in float32 format.
    """
    total_samples, n_channels = raw_data.shape
    n_segments = (total_samples + segment_len - 1) // segment_len

    baselines = np.zeros((n_channels, n_segments), dtype=np.float32)

    for seg_idx in range(n_segments):
        start = seg_idx * segment_len
        end = min(start + segment_len, total_samples)
        segment = raw_data[start:end, :]  # Shape: [S, C]

        if segment.shape[0] < 2:
            baselines[:, seg_idx] = 0
            continue

        # Compute absolute derivative
        diff_segment = np.abs(np.diff(segment, axis=0))  # Shape: [S-1, C]
        # Pad to match original length
        diff_segment = np.vstack([diff_segment, diff_segment[-1]])

        # Mask: keep only low-derivative points
        flat_mask = diff_segment < diff_thresh  # Shape: [S, C]

        # Apply mask and compute trimmed mean per channel
        for c in range(n_channels):
            flat_vals = segment[flat_mask[:, c], c].astype(np.float32)
            if len(flat_vals) > 0:
                baselines[c, seg_idx] = trim_mean(flat_vals, proportiontocut=trim_fraction)
            else:
                baselines[c, seg_idx] = 0  # Fallback

    return baselines


def subtract_segment_baselines_int16(raw_data: np.ndarray,
                                     baselines_f32: np.ndarray,
                                     segment_len: int = 100_000) -> None:
    """
    In-place baseline removal for int16 raw traces.

    Parameters
    ----------
    raw_data : np.ndarray, shape (T, C)
        The entire int16 recording in RAM. This array is modified in-place.
    baselines_f32 : np.ndarray, shape (C, n_segments)
        The baseline values from compute_baselines_int16_deriv_robust.
    segment_len : int
        The same segment length used to compute the baselines.
    """

    T, C = raw_data.shape
    C_b, n_seg = baselines_f32.shape
    if C_b != C:
        raise ValueError("Channel count mismatch between raw_data and baselines")

    # Quantise baselines once
    baselines_i16 = np.rint(baselines_f32).astype(np.int16)

    for seg_idx in range(n_seg):
        start = seg_idx * segment_len
        end   = min(start + segment_len, T) # Handle last partial segment

        # Broadcast-subtract:  [end-start, C]  -=  [C]
        raw_data[start:end, :] -= baselines_i16[:, seg_idx]