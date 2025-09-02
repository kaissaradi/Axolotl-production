# axolotl/comparison.py
"""
Functions for comparing electrical images (EIs) using similarity metrics
like cosine similarity, lag-tolerant alignment, and subtraction.
"""
import numpy as np
from scipy.signal import correlate
from scipy.interpolate import interp1d

def compare_eis(eis, ei_template=None, max_lag=30, thr=30.0):
    """
    If `ei_template` is None  → full pair-wise similarity matrix  [k,k].
    If `ei_template` given     → column vector of similarities  [k,1].
    Similarity is cosine over signal channels only.
    """
    if isinstance(eis, list):
        eis = np.stack(eis, axis=0)
    k, C, T = eis.shape
    ptp = np.ptp(eis, axis=2)

    if ei_template is None:
        sim = np.zeros((k, k), dtype=np.float32)
        for i in range(k):
            ei_i = eis[i]
            dom_i = np.argmax(ptp[i])
            trace_i = ei_i[dom_i]
            for j in range(i, k):
                ei_j = eis[j]
                dom_j = np.argmax(ptp[j])
                trace_j = ei_j[dom_j]
                lags = np.arange(-max_lag, max_lag + 1)
                xc = correlate(trace_i, trace_j, mode="full", method="auto")
                center = len(xc) // 2
                shift = lags[np.argmax(xc[center - max_lag:center + max_lag + 1])]
                ei_j_aligned = np.roll(ei_j, shift, axis=1)
                mask = (ptp[i] > thr) | (ptp[j] > thr)
                if mask.any():
                    a = ei_i[mask].ravel()
                    b = ei_j_aligned[mask].ravel()
                    val = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
                else:
                    val = 0.0
                sim[i, j] = sim[j, i] = val
        return sim
    else:
        sim = np.zeros((k, 1), dtype=np.float32)
        ptp_t = np.ptp(ei_template, axis=1)
        dom_t = int(np.argmax(ptp_t))
        trace_t = ei_template[dom_t]
        for i in range(k):
            ei_i = eis[i]
            dom_i = int(np.argmax(ptp[i]))
            trace_i = ei_i[dom_i]
            lags = np.arange(-max_lag, max_lag + 1)
            xc = correlate(trace_i, trace_t, mode="full", method="auto")
            center = len(xc) // 2
            shift = lags[np.argmax(xc[center - max_lag:center + max_lag + 1])]
            ei_t_aligned = np.roll(ei_template, shift, axis=1)
            mask = (ptp[i] > thr) | (ptp_t > thr)
            if mask.any():
                a = ei_i[mask].ravel()
                b = ei_t_aligned[mask].ravel()
                sim[i, 0] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            else:
                sim[i, 0] = 0.0
        return sim


def sub_sample_align_ei(ei_template, ei_candidate, ref_channel, upsample=10, max_shift=2.0):
    """
    Align ei_candidate to ei_template using sub-sample alignment on the reference channel.
    """
    C, T = ei_template.shape
    assert ei_candidate.shape == (C, T), "Shape mismatch"

    t = np.arange(T)
    t_interp = np.linspace(0, T - 1, T * upsample)

    interp_template = interp1d(t, ei_template[ref_channel], kind='cubic', bounds_error=False, fill_value=0.0)
    interp_candidate = interp1d(t, ei_candidate[ref_channel], kind='cubic', bounds_error=False, fill_value=0.0)

    template_highres = interp_template(t_interp)
    candidate_highres = interp_candidate(t_interp)

    full_corr = correlate(candidate_highres, template_highres, mode='full')
    lags = np.arange(-len(candidate_highres) + 1, len(template_highres))
    center = len(full_corr) // 2
    lag_window = int(max_shift * upsample)
    search_range = slice(center - lag_window, center + lag_window + 1)

    best_lag_index = np.argmax(full_corr[search_range])
    fractional_shift = lags[search_range][best_lag_index] / upsample

    aligned_candidate = np.zeros_like(ei_candidate)
    for ch in range(C):
        interp_func = interp1d(t, ei_candidate[ch], kind='cubic', bounds_error=False, fill_value=0.0)
        shifted_time = t + fractional_shift
        aligned_candidate[ch] = interp_func(shifted_time)

    return aligned_candidate, fractional_shift


def compare_ei_subtraction(ei_a, ei_b, max_lag=3, p2p_thresh=30.0):
    """
    Compare two EIs using subtraction and cosine similarity.
    """
    C, T = ei_a.shape
    assert ei_b.shape == (C, T), "EIs must have same shape"

    ref_chan = np.argmax(np.max(np.abs(ei_a), axis=1))
    aligned_b, fractional_shift = sub_sample_align_ei(ei_template=ei_a, ei_candidate=ei_b, ref_channel=ref_chan, upsample=10, max_shift=max_lag)

    p2p_a = np.ptp(ei_a, axis=1)
    good_channels = np.where(p2p_a > p2p_thresh)[0]

    per_channel_residuals = []
    per_channel_cosine_sim = []
    all_residuals = []

    for ch in good_channels:
        a = ei_a[ch]
        b = aligned_b[ch]
        mask = np.abs(a) > 0.1 * np.max(np.abs(a))
        if not np.any(mask):
            continue
        
        a_masked, b_masked = a[mask], b[mask]
        residual = b_masked - a_masked
        per_channel_residuals.append(np.mean(residual))
        all_residuals.extend(residual)
        
        dot = np.dot(a_masked, b_masked)
        norm_product = np.linalg.norm(a_masked) * np.linalg.norm(b_masked) + 1e-8
        per_channel_cosine_sim.append(dot / norm_product)

    if not all_residuals:
        return {'mean_residual': np.nan, 'max_abs_residual': np.nan, 'good_channels': good_channels,
                'per_channel_residuals': per_channel_residuals, 'per_channel_cosine_sim': per_channel_cosine_sim,
                'fractional_shift': fractional_shift, 'p2p_a': p2p_a}
    
    mean_residual = np.mean(all_residuals)
    max_abs_residual = np.max(np.abs(all_residuals))

    return {
        'mean_residual': mean_residual,
        'max_abs_residual': max_abs_residual,
        'good_channels': good_channels,
        'per_channel_residuals': per_channel_residuals,
        'per_channel_cosine_sim': per_channel_cosine_sim,
        'fractional_shift': fractional_shift,
        'p2p_a': p2p_a
    }