# axolotl/subtraction.py
"""
Functions for subtracting found neuron templates from the raw data trace (peeling).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from numba import njit

def subtract_pca_cluster_means_ram(snippets, baselines, spike_times, segment_len=100_000, n_clusters=5, offset_window=(-5,10)):
    """
    Subtracts PCA-clustered mean waveforms from baseline-corrected spike snippets for a single channel.

    Parameters
    ----------
    snippets : np.ndarray, shape (n_spikes, snip_len)
        Array for a single channel.
    baselines : np.ndarray, shape (n_segments,)
        Array of mean baseline per segment for this channel.
    spike_times : np.ndarray, shape (n_spikes,)
        Array of spike times (in samples).
    segment_len : int
        Segment size used for baseline estimation.
    n_clusters : int
        Number of PCA/k-means clusters to use.
    offset_window: tuple
        Window relative to the peak for PCA and subtraction.

    Returns
    -------
    residuals : np.ndarray, shape (n_spikes, snip_len)
        int16 array of subtracted residuals with baseline added back.
    scale_factors : np.ndarray, shape (n_spikes,)
        float32 array of amplitude scaling per spike.
    cluster_ids : np.ndarray, shape (n_spikes,)
        int32 array of cluster IDs.
    """
    # --- Baseline subtraction ---
    segment_ids = spike_times // segment_len
    segment_ids = np.clip(segment_ids, 0, len(baselines) - 1)
    baseline_per_spike = baselines[segment_ids][:, np.newaxis]
    snippets_bs = snippets - baseline_per_spike

    # --- Template and window ---
    template = np.mean(snippets_bs, axis=0)
    neg_peak_idx = np.argmin(template)
    w_start = max(0, neg_peak_idx + offset_window[0])
    w_end = min(snippets.shape[1], neg_peak_idx + offset_window[1])
    window = slice(w_start, w_end)

    # --- PCA and clustering ---
    pca = PCA(n_components=5)
    # Use only the window for PCA
    reduced = pca.fit_transform(snippets_bs[:, window])
    cluster_ids = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit_predict(reduced)

    # --- Subtract cluster means (only in window) ---
    residuals = np.copy(snippets_bs)
    scale_factors = np.ones(snippets_bs.shape[0], dtype=np.float32) # Placeholder
    
    for c in range(n_clusters):
        idx = np.where(cluster_ids == c)[0]
        if len(idx) == 0:
            continue
        cluster_mean = np.mean(snippets_bs[idx, window], axis=0)
        residuals[idx, window] -= cluster_mean

    # --- Restore baseline, clip, convert ---
    residuals += baseline_per_spike
    residuals = np.clip(residuals, -32768, 32767).astype(np.int16)

    return residuals, scale_factors, cluster_ids


@njit(cache=True)
def _apply_residuals_channel(raw_data_ch, residuals, write_locs):
    """Numba-accelerated helper to apply residuals for a SINGLE channel."""
    total_samples = len(raw_data_ch)
    n_spikes = len(write_locs)
    snip_len = residuals.shape[1]
    
    for i in range(n_spikes):
        loc = write_locs[i]
        end = loc + snip_len
        if loc >= 0 and end <= total_samples:
            raw_data_ch[loc:end] = residuals[i, :]

def apply_residuals(
    raw_data: np.ndarray,
    residual_snips_per_channel: dict,
    write_locs: np.ndarray,
    selected_channels: np.ndarray,
    total_samples: int,
    is_ram: bool = True
):
    """
    Applies residual snippets to time-major data by calling a fast Numba kernel.
    """
    if not is_ram:
        raise NotImplementedError("Disk-based modification is not supported.")

    # This loop runs in normal Python
    for ch in selected_channels:
        if ch not in residual_snips_per_channel:
            continue
            
        residuals = residual_snips_per_channel[ch]

        if residuals.shape[0] != len(write_locs):
            raise ValueError(f"Mismatch between residuals and write_locs for channel {ch}")

        # Call the fast Numba kernel for the actual data modification
        _apply_residuals_channel(raw_data[:, ch], residuals, write_locs)


def subtract_scaled_template_ram(snippets, template):
    """
    Subtracts a scaled version of a template from a set of snippets.
    This is a simplified, robust subtraction function.

    Parameters
    ----------
    snippets : np.ndarray, shape (n_spikes, snip_len)
        Array for a single channel.
    template : np.ndarray, shape (snip_len,)
        Array of the template to subtract.

    Returns
    -------
    residuals : np.ndarray, shape (n_spikes, snip_len)
        int16 array of subtracted residuals.
    """
    n_spikes, snip_len = snippets.shape
    
    template_f32 = template.astype(np.float32)
    snippets_f32 = snippets.astype(np.float32)

    dot_product = np.dot(snippets_f32, template_f32)
    template_norm_sq = np.dot(template_f32, template_f32) + 1e-6
    
    scale_factors = dot_product / template_norm_sq
    scale_factors = np.clip(scale_factors, 0.5, 2.0)
    
    residuals_f32 = snippets_f32 - scale_factors[:, np.newaxis] * template_f32
    
    residuals = np.clip(residuals_f32, -32768, 32767).astype(np.int16)
    
    return residuals