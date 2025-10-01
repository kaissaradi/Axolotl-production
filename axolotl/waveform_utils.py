# axolotl/waveform_utils.py
"""
Core utility functions for operating on snippets, waveforms, and spike times.
"""
import numpy as np
from scipy.signal import find_peaks, correlate
from scipy.optimize import lsq_linear, nnls
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde, trim_mean
from sklearn.mixture import GaussianMixture
from numba import njit, prange

def extract_snippets_fast_ram(
    raw_data: np.ndarray,
    spike_times: np.ndarray,
    window: tuple[int, int],
    selected_channels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts snippets from time-major raw_data around spike_times without a Python loop.

    Parameters
    ----------
    raw_data : np.ndarray, shape (T, C)
        The raw data array (int16 or float).
    spike_times : np.ndarray, shape (N,)
        Array of spike sample indices.
    window : tuple
        A tuple of (pre, post) samples for the snippet window.
    selected_channels : np.ndarray
        An array of channel indices to extract.

    Returns
    -------
    tuple
        - np.ndarray, shape (K, L, N_valid): Snippets as float32.
        - np.ndarray, shape (N_valid,): The spike times that were within bounds.
    """
    pre, post = window
    win_len = post - pre + 1
    total_samples, _ = raw_data.shape

    valid_mask = (spike_times + pre >= 0) & (spike_times + post < total_samples)
    valid_times = spike_times[valid_mask]
    n_valid = valid_times.size
    n_channels_selected = len(selected_channels)

    if n_valid == 0:
        return np.empty((n_channels_selected, win_len, 0), np.float32), valid_times

    offsets = np.arange(pre, post + 1, dtype=np.int64)
    time_indices = (valid_times[:, None] + offsets[None, :]).ravel()

    snippets = raw_data[time_indices[:, None], selected_channels]
    snippets = snippets.astype(np.float32, copy=False)
    snippets = snippets.reshape(n_valid, win_len, n_channels_selected).transpose(2, 1, 0)

    return snippets, valid_times

@njit(fastmath=True, cache=True)
def _manual_xcorr(a, b):
    """A Numba-friendly cross-correlation for 'same' mode."""
    len_a = len(a)
    len_b = len(b)
    len_corr = len_a - len_b + 1
    correlation = np.zeros(len_corr)
    for i in range(len_corr):
        correlation[i] = np.dot(a[i:i+len_b], b)
    return correlation

def estimate_lags_by_xcorr_ram(
    snippets: np.ndarray,
    peak_channel_idx: int,
    window: tuple,
    max_lag: int
) -> np.ndarray:
    """
    Estimates temporal lags for spikes by cross-correlating each snippet
    with the mean waveform on the peak channel. ACCELERATED WITH NUMBA.
    """
    # This part runs in regular Python
    mean_waveform = np.mean(snippets[:, peak_channel_idx, :], axis=0)
    peak_idx = np.argmin(mean_waveform)
    start, end = peak_idx + window[0], peak_idx + window[1]

    template = mean_waveform[start:end]
    data = snippets[:, peak_channel_idx, start:end]

    # Call the fast, Numba-compiled inner loop
    lags = _estimate_lags_numba(data, template, max_lag)
    return lags

@njit(fastmath=True, cache=True, parallel=True)
def _estimate_lags_numba(data, template, max_lag):
    """Inner loop for lag estimation, compiled by Numba."""
    n_spikes = data.shape[0]
    lags = np.zeros(n_spikes, dtype=np.int32)
    # Numba can run this loop in parallel for extra speed
    for i in prange(n_spikes):
        # This is a simplified, manual correlation
        correlation = _manual_xcorr(data[i, :], template)
        # Center the lag calculation
        lag = np.argmax(correlation) - (len(correlation) // 2)
        lags[i] = max(-max_lag, min(lag, max_lag))
    return lags

def check_2d_gap_peaks_valley(pcs, angle_step=10, min_valley_frac=0.35):
    """
    Projects 2D data along multiple angles, finds the projection with the
    most prominent valley between two peaks in its histogram, and returns a
    split based on that valley.
    """
    from diptest import diptest
    
    best_p, best_proj, best_angle = 1.0, None, 0
    angles = np.deg2rad(np.arange(0, 180, angle_step))

    for theta in angles:
        proj = pcs[:, 0] * np.cos(theta) + pcs[:, 1] * np.sin(theta)
        _, p = diptest(proj)
        if p < best_p:
            best_p, best_proj, best_angle = p, proj, np.rad2deg(theta)

    if best_proj is None:
        return None

    kde = gaussian_kde(best_proj)
    x_grid = np.linspace(best_proj.min(), best_proj.max(), 512)
    density = kde(x_grid)
    
    peaks, _ = find_peaks(density)
    valleys, _ = find_peaks(-density)

    if len(peaks) < 2 or len(valleys) == 0:
        return None

    sorted_peaks = peaks[np.argsort(-density[peaks])[:2]]
    if len(sorted_peaks) < 2:
        return None
        
    p1, p2 = min(sorted_peaks), max(sorted_peaks)
    
    valley_candidates = [v for v in valleys if p1 < v < p2]
    if not valley_candidates:
        return None
        
    valley_idx = valley_candidates[np.argmin(density[valley_candidates])]
    
    valley_height = density[valley_idx]
    min_peak_height = min(density[p1], density[p2])

    if valley_height < min_valley_frac * min_peak_height:
        split_point = x_grid[valley_idx]
        return best_proj < split_point, best_proj >= split_point

    return None

def fit_two_templates_bounded(y, TA, TB, lower=0.75, upper=1.25):
    """Fit two templates to a signal with bounded coefficients."""
    X = np.vstack((TA, TB)).T
    y64 = y.astype(np.float64)
    result = lsq_linear(X, y64, bounds=(lower, upper), lsmr_tol='auto', verbose=0)
    alpha, beta = result.x
    rnorm = np.linalg.norm(y - X @ result.x)
    return alpha, beta, rnorm

def fit_two_templates(y, TA, TB_shift):
    """Fit two templates to a signal using non-negative least squares."""
    X = np.vstack((TA, TB_shift)).T
    coeffs, rnorm = nnls(X, y)
    return coeffs[0], coeffs[1], rnorm

def build_channel_weights_twoEI(
    snipsA_good, snipsB_good, eiA, eiB, p2p_thresh=30.0, rel_mask=0.01
):
    """Builds channel weights based on P2P amplitude and noise variance."""
    C, T = eiA.shape
    p2p_A = np.ptp(eiA, axis=1)
    p2p_B = np.ptp(eiB, axis=1)
    keep = (p2p_A >= p2p_thresh) | (p2p_B >= p2p_thresh)
    sel = np.where(keep)[0]
    if sel.size == 0:
        raise ValueError("No channels pass p2p_thresh")

    mask = np.zeros((sel.size, T), dtype=bool)
    for k, c in enumerate(sel):
        tmpl = eiA[c] if p2p_A[c] >= p2p_B[c] else eiB[c]
        p2p = max(p2p_A[c], p2p_B[c])
        mask[k] = np.abs(tmpl) >= rel_mask * p2p

    sigma2 = np.empty(sel.size, dtype=np.float32)
    for k, c in enumerate(sel):
        res = snipsA_good[c] - eiA[c][:, None] if p2p_A[c] >= p2p_B[c] else snipsB_good[c] - eiB[c][:, None]
        res_masked = res[mask[k]]
        mad = np.median(np.abs(res_masked))
        sigma = mad / 0.6745 + 1e-6
        sigma2[k] = sigma * sigma

    p2p_sel = np.maximum(p2p_A[sel], p2p_B[sel]).astype(np.float32)
    weights = p2p_sel / sigma2
    weights /= weights.mean()

    return weights.astype(np.float32), sel.astype(np.int32)

@njit(fastmath=True, cache=True)
def _roll_pad(arr, lag):
    """Helper for rolling and padding arrays within Numba-compiled functions."""
    C, T = arr.shape
    out = np.zeros_like(arr)
    if lag > 0:
        out[:, lag:] = arr[:, :T-lag]
    elif lag < 0:
        out[:, :T+lag] = arr[:, -lag:]
    else:
        out[:] = arr
    return out

