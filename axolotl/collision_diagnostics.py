"""axolotl/collision_diagnostics.py

Diagnostic, analysis, and plotting helpers extracted from collision.py.
Core pipeline functions remain in collision.py.

Imports score_active_set from collision for marginal_gain.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.signal import correlate, correlation_lags
from types import SimpleNamespace

from .plotting import plot_ei_waveforms
from .collision import score_active_set


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def delta_rms(x, y, weights):
    rms_raw = np.sqrt((x**2).mean(axis=1))        # [C]
    rms_res = np.sqrt(((x - y)**2).mean(axis=1))  # [C]
    return np.sum(weights * (rms_res - rms_raw))


def best_shift_ei(trace_rw, trace_ei, max_shift=40):
    """
    Find the best right shift (0…max_shift) to apply to EI to match trace_rw.

    Returns
    -------
    best_lag : int  (0…max_shift)
    """
    L = len(trace_rw)
    scores = []
    for lag in range(max_shift + 1):
        ei_shifted = np.zeros_like(trace_ei)
        ei_shifted[lag:] = trace_ei[:L - lag]
        s = np.dot(trace_rw, ei_shifted)
        scores.append((lag, s))
    best_lag = max(scores, key=lambda x: x[1])[0]
    return best_lag


def _peak_channel(ei):
    """Index of channel with largest |P2P| in EI."""
    return int(np.argmax(ei.ptp(axis=1)))


def _best_lag(raw_chan, ei_chan, peak_sample=40, max_lag=6):
    """
    Dot-product lag search around ±max_lag
    so that the EI peak ends near `peak_sample`.
    """
    xcor = correlate(raw_chan, ei_chan, mode='full')
    lags = np.arange(-len(raw_chan) + 1, len(raw_chan))
    lag_raw = lags[np.argmax(xcor)]

    p_idx = np.argmax(np.abs(ei_chan))
    base = len(raw_chan) // 2 - p_idx
    lag_low = base - max_lag
    lag_hi = base + max_lag
    return int(np.clip(lag_raw, lag_low, lag_hi))


def marginal_gain(active_dict,
                  union_chans,
                  raw_local,
                  unit_info,
                  p2p_all,
                  *,
                  mask_thr=5.0,
                  beta=0.5,
                  max_lag=6,
                  peak_sample=40,
                  score_fn=score_active_set):
    """
    Return (uid, best_lag, gain) of the *first* unit that improves
    Δ-RMS when added to `active_dict`.  If none do, return None.
    """
    from collections import OrderedDict

    base_score = score_fn(active_dict, union_chans,
                          raw_local, unit_info, p2p_all,
                          mask_thr=mask_thr, beta=beta)

    for uid in unit_info.keys():
        if uid in active_dict:
            continue

        ei = unit_info[uid]["ei"]
        pch = _peak_channel(ei)

        if pch not in union_chans:
            continue
        c_idx = union_chans.index(pch)

        lag = _best_lag(raw_local[c_idx], ei[pch],
                        peak_sample=peak_sample,
                        max_lag=max_lag)

        trial = OrderedDict(active_dict)
        trial[uid] = lag

        new_score = score_fn(trial, union_chans,
                             raw_local, unit_info, p2p_all,
                             mask_thr=mask_thr, beta=beta)

        gain = new_score - base_score
        if gain > 0:
            print(uid, lag, gain)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Bimodality / clustering
# ─────────────────────────────────────────────────────────────────────────────

def per_channel_gmm_bimodality_simple(ei, snips, n_top=5, min_sep=2.0,
                                      include_ref40=True, win40=3, min_cluster_size=20):
    """
    For each channel, evaluate 2-GMM separation at multiple candidate times:
      - channel's own t_peak = argmax |ei[c]|
      - optional window [40-win40, 40+win40]
    Return top (sep, c, t, vmin, vmax).
    """
    C, T, N = snips.shape
    results = []
    for c in range(C):
        t_peak = int(np.argmax(np.abs(ei[c])))
        cand_times = {t_peak}
        if include_ref40:
            for d in range(-win40, win40 + 1, 2):
                t = t_peak + d
                if 0 <= t < T:
                    cand_times.add(t)

        best = None
        for t in cand_times:
            v = snips[c, t, :].reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
            try:
                gmm.fit(v)
            except Exception:
                continue
            labels = gmm.predict(v)
            counts = np.bincount(labels, minlength=2)
            if counts.min() < min_cluster_size:
                continue

            mu1, mu2 = gmm.means_.flatten()
            std1 = np.sqrt(gmm.covariances_[0][0, 0])
            std2 = np.sqrt(gmm.covariances_[1][0, 0])
            sep = abs(mu1 - mu2) / np.sqrt(0.5 * (std1**2 + std2**2))
            if sep >= min_sep and (best is None or sep > best[0]):
                best = (sep, c, t, float(v.min()), float(v.max()))
        if best is not None:
            results.append(best)

    results.sort(key=lambda x: -x[0])
    return results[:n_top]


def per_channel_hist_otsu_candidates(
    ei: np.ndarray,
    snips: np.ndarray,
    n_top: int = 5,
    *,
    sim_thr: float = 0.90,
    win40: int = 2,
    bins: int = 64,
    min_cluster_size: int = 10
):
    C, T, N = snips.shape
    results = []

    for c in range(C):
        t_peak = int(np.argmax(np.abs(ei[c])))
        cand_times = {t_peak}
        for d in range(-win40, win40 + 1):
            t = t_peak + d
            if 0 <= t < T:
                cand_times.add(t)

        best = None

        for t in cand_times:
            v = snips[c, t, :].astype(np.float32)

            p1, p99 = np.percentile(v, [1, 99])
            if not np.isfinite(p1) or not np.isfinite(p99) or p1 >= p99:
                continue
            edges = np.linspace(p1, p99, bins + 1)
            hist, _ = np.histogram(v, bins=edges)
            if hist.sum() == 0:
                continue

            p = hist.astype(np.float64) / hist.sum()
            mids = 0.5 * (edges[:-1] + edges[1:])
            w0 = np.cumsum(p)
            mu0 = np.cumsum(p * mids)
            muT = mu0[-1]
            w1 = 1.0 - w0
            denom = (w0 * w1) + 1e-12
            sigma_b2 = ((muT * w0 - mu0) ** 2) / denom

            if sigma_b2.size < 3:
                continue
            k = int(np.nanargmax(sigma_b2[1:-1])) + 1
            thr = mids[k]

            mask_lo = v <= thr
            n_lo = int(mask_lo.sum())
            n_hi = int((~mask_lo).sum())
            if min(n_lo, n_hi) < min_cluster_size:
                continue

            wav_lo = np.mean(snips[c, :, mask_lo], axis=1)
            wav_hi = np.mean(snips[c, :, ~mask_lo], axis=1)
            amp_lo = float(wav_lo.max() - wav_lo.min())
            amp_hi = float(wav_hi.max() - wav_hi.min())
            score_amp = max(amp_lo, amp_hi)

            if (best is None) or (score_amp < best[0]):
                best = (score_amp, t, thr, n_lo, n_hi, mask_lo.copy())

        if best is None:
            continue

        score_amp, t_best, thr_best, n_lo, n_hi, mask_lo_best = best
        results.append((score_amp, c, t_best, float(thr_best), n_lo, n_hi))

    results.sort(key=lambda x: x[3])
    return results[:n_top]


def bimodality_probe(snips_cand, c_best, t_best, SHOW_DIAGNOSTICS,
                     offsets=(-5, -3, -1, 1, 3, 5, 7, 9, 11)):
    T, N = snips_cand.shape[1], snips_cand.shape[2]
    reports = []
    for d in offsets:
        t = t_best + d
        if t < 0 or t >= T:
            continue
        v = snips_cand[c_best, t, :].astype(float)
        X = v.reshape(-1, 1)
        g1 = GaussianMixture(1, reg_covar=1e-6, random_state=0).fit(X)
        g2 = GaussianMixture(2, reg_covar=1e-6, random_state=0).fit(X)
        bic1 = float(g1.bic(X)); bic2 = float(g2.bic(X)); db = bic1 - bic2
        idx = np.argsort(g2.means_.ravel())
        mu = g2.means_.ravel()[idx]
        sd = np.sqrt(g2.covariances_.reshape(-1))[idx]
        wt = g2.weights_[idx]
        dpr = float(abs(mu[1] - mu[0]) / (np.sqrt(0.5 * (sd[0]**2 + sd[1]**2)) + 1e-12))
        reports.append(dict(offset=d, sample=int(t), N=int(N),
                            bic1=bic1, bic2=bic2, delta_bic=float(db),
                            dprime=dpr, means=mu.astype(float),
                            stds=sd.astype(float), weights=wt.astype(float),
                            sizes=(wt * N).round().astype(int)))
    if not reports:
        print(f"\nBIMODALITY PROBE: c={c_best}, t_best={t_best} → no valid offsets in {list(offsets)}")
        return None
    best = max(reports, key=lambda r: (r['delta_bic'], r['dprime']))
    if SHOW_DIAGNOSTICS:
        mu = ", ".join(f"{m:.2f}" for m in best['means'])
        sd = ", ".join(f"{s:.2f}" for s in best['stds'])
        wt = ", ".join(f"{w:.2f}" for w in best['weights'])
        sz = ", ".join(str(int(s)) for s in best['sizes'])
        print("\nBIMODALITY PROBE (best channel, off-peak)")
        print(f"  c={c_best}  t_best={t_best}  tested={list(offsets)}")
        print(f"  selected t={best['sample']} (off {best['offset']}), N={best['N']}")
        print(f"  BIC1={best['bic1']:.2f}  BIC2={best['bic2']:.2f}  ΔBIC={best['delta_bic']:.2f}  d′={best['dprime']:.2f}")
        print(f"  means=[{mu}]  stds=[{sd}]  weights=[{wt}]  sizes=[{sz}]")
    return best


def recursive_bimodal_split(
    snips_pool, cand_mask, *,
    channel_of_interest, c_best, t_best,
    cfg, SHOW_DIAGNOSTICS,
    max_splits=4,
    size_mult=3,
    bic_thr=300.0, dprime_thr=5.0
):
    """
    Repeatedly check bimodality and split the picked cluster on one channel/time,
    until it is no longer bimodal (on both probes) or the selected subcluster is too small.

    Returns:
        cand_mask   : updated boolean mask on the CURRENT pool (global indices)
        ei_cand     : median EI of the final picked cluster
        n_splits    : number of successful splits performed
        last_probe  : ('c','t','metric_val') tuple of the split that happened last (or None)
    """
    n_splits = 0
    last_probe = None

    probes = [(int(channel_of_interest), 40)]
    if int(c_best) != int(channel_of_interest):
        probes.append((int(c_best), int(t_best)))

    while True:
        snips_cand = snips_pool[:, :, cand_mask]
        n_cur = snips_cand.shape[2]
        if n_cur < cfg.MIN_CLUSTER_SIZE:
            if SHOW_DIAGNOSTICS:
                print(f"RECURSION STOP: cluster too small for any further work (n={n_cur} < {cfg.MIN_CLUSTER_SIZE})")
            break

        split_happened = False
        for (c_probe, t_probe) in probes:
            best = bimodality_probe(snips_cand, c_probe, t_probe, SHOW_DIAGNOSTICS)
            if best is None:
                continue
            if (best['delta_bic'] <= bic_thr) and (best['dprime'] <= dprime_thr):
                continue

            t_sel = int(best['sample'])
            v = snips_cand[c_probe, t_sel, :].astype(float).reshape(-1, 1)

            g2 = GaussianMixture(n_components=2, reg_covar=1e-6, random_state=0).fit(v)
            mu = g2.means_.ravel()
            labels = g2.predict(v)

            try_order = np.argsort(np.abs(mu))[::-1]
            idx_cand_loc = np.where(cand_mask)[0]
            picked = None
            thr = size_mult * cfg.MIN_CLUSTER_SIZE

            for k in try_order:
                submask_local = (labels == k)
                n_sub = int(submask_local.sum())
                if n_sub >= thr:
                    new_mask = np.zeros_like(cand_mask, dtype=bool)
                    new_mask[idx_cand_loc[submask_local]] = True
                    if SHOW_DIAGNOSTICS:
                        print(
                            f"SPLIT-GATE TRIGGERED (recursive) → "
                            f"c={c_probe}, t={t_sel}, picked comp {k} |μ|={abs(mu[k]):.2f}, "
                            f"size={n_sub}/{idx_cand_loc.size} (thr={thr})"
                        )
                    cand_mask = new_mask
                    n_splits += 1
                    last_probe = (c_probe, t_sel, float(best['delta_bic']))
                    split_happened = True
                    break
                else:
                    if SHOW_DIAGNOSTICS:
                        print(f"  split candidate comp {k} too small: {n_sub} < {thr} (skipping)")

            if split_happened:
                break

        if not split_happened:
            if SHOW_DIAGNOSTICS:
                print("RECURSION STOP: no probe shows strong bimodality or subclusters too small.")
            break

        if n_splits >= max_splits:
            if SHOW_DIAGNOSTICS:
                print(f"RECURSION STOP: reached max_splits={max_splits}.")
            break

    ei_cand = median_ei_adaptive(snips_pool[:, :, cand_mask])
    return cand_mask, ei_cand, n_splits, last_probe


# ─────────────────────────────────────────────────────────────────────────────
# Baseline / similarity
# ─────────────────────────────────────────────────────────────────────────────

def compute_global_baseline_mean(ei,
                                 early=20,
                                 late=20,
                                 cap_val=50,
                                 make_plots=False,
                                 baseline_thresh=10):
    """
    Fast vectorised baseline-quality metric.
    Returns global_baseline_mean (float).
    """
    n_ch, n_t = ei.shape

    p2p = np.ptp(ei, axis=1)

    pos_idx = np.argmax(ei, axis=1)
    neg_idx = np.argmin(ei, axis=1)
    pos_val_abs = ei[np.arange(n_ch), pos_idx]
    neg_val_abs = np.abs(ei[np.arange(n_ch), neg_idx])

    use_pos = pos_val_abs > neg_val_abs
    peak_idx = np.where(use_pos, pos_idx, neg_idx)

    early_ok = peak_idx >= early
    late_ok = peak_idx < (n_t - late)

    early_block_mean = ei[:, :early].mean(axis=1)
    early_block_max = ei[:, :early].max(axis=1)
    late_block_mean = ei[:, -late:].mean(axis=1)
    late_block_max = ei[:, -late:].max(axis=1)

    early_mean = np.where(early_ok, early_block_mean, np.nan)
    early_max = np.where(early_ok, early_block_max, np.nan)
    late_mean = np.where(late_ok, late_block_mean, np.nan)
    late_max = np.where(late_ok, late_block_max, np.nan)

    abs_early_mean = np.abs(early_mean)
    abs_late_mean = np.abs(late_mean)

    bad_baseline_ch = (abs_early_mean > baseline_thresh) | (abs_late_mean > baseline_thresh)
    n_bad_channels = np.count_nonzero(bad_baseline_ch)

    baseline_means_all = np.concatenate([abs_early_mean[np.isfinite(early_mean)],
                                         abs_late_mean[np.isfinite(late_mean)]])
    baseline_max_all = np.concatenate([np.abs(early_max[np.isfinite(early_max)]),
                                       np.abs(late_max[np.isfinite(late_max)])])

    global_baseline_mean = baseline_means_all.mean()
    global_baseline_max_mean = baseline_max_all.mean()

    if make_plots:
        fig, axs = plt.subplots(2, 2, figsize=(18, 6), sharex=True)
        axs = axs.ravel()

        axs[0].plot(p2p, color='black')
        axs[0].set_title('P2P amplitude')
        axs[0].set_ylabel('Amplitude')

        axs[1].plot(peak_idx, color='blue')
        axs[1].set_title('Location of dominant peak')
        axs[1].set_ylabel('Sample')

        axs[2].plot(early_mean, label='Early mean', color='green')
        axs[2].plot(late_mean, label='Late mean', color='orange')
        axs[2].set_title('Baseline mean')
        axs[2].set_ylim(-cap_val, cap_val)
        axs[2].legend()

        axs[3].plot(early_max, label='Early max', color='green')
        axs[3].plot(late_max, label='Late max', color='orange')
        axs[3].set_title('Baseline max')
        axs[3].set_ylim(-cap_val, cap_val)
        axs[3].legend()

        for ax in axs:
            ax.set_xlabel('Channel')
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    return global_baseline_mean


def ei_similarity(ei_a, ei_b):
    """
    Cosine similarity – range [0..1]. Higher ⇒ more alike.
    """
    a = ei_a.ravel(); b = ei_b.ravel()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def median_ei_adaptive(snips, base=500):
    """
    Compute EI as the median of an adaptively sub-sampled set of spikes.

    snips : ndarray  [C, T, N]
    base  : int — use every spike when N ≤ base; stride increases by 1
            for every additional `base` spikes.

    Returns
    -------
    ei_med : ndarray  [C, T]
    """
    N = snips.shape[2]
    stride = 1 + (N - 1) // base
    ei_med = np.median(snips[:, :, ::stride], axis=2)
    return ei_med.astype(snips.dtype, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# Recursive cluster validation
# ─────────────────────────────────────────────────────────────────────────────

def verify_clusters_recursive(spike_times,
                               snips_raw,
                               params,
                               target_chan,
                               abs_idx=None,
                               depth=0):
    """
    Returns list of dicts with keys:
        'inds'  – indices into *spike_times*
        'ei'    – median EI [C, T]
    """
    import time

    if abs_idx is None:
        abs_idx = np.arange(spike_times.size)

    min_spikes = params.get('min_spikes', 10)
    baseline_cut = params.get('baseline_cut', 2.0)
    ei_sim_thr = params.get('ei_sim_threshold', 0.9)
    k_start = params.get('k_start', 6)
    k_refine = params.get('k_refine', 3)

    N = spike_times.size
    if N < min_spikes:
        return []

    print(f"Depth: {depth}")

    start = time.perf_counter()
    ei_parent = median_ei_adaptive(snips_raw)
    peak_ch = np.argmin(ei_parent.min(axis=1))

    ei_parent = np.ascontiguousarray(ei_parent, dtype=np.float32)

    gbm_parent = compute_global_baseline_mean(ei_parent)
    print(f"          parent gmb: {gbm_parent}")

    if peak_ch != target_chan or gbm_parent >= baseline_cut:
        return []

    k_split = k_start if depth == 0 else k_refine
    k_split = max(2, min(k_split, N // min_spikes))
    if k_split < 2:
        return [{'inds': abs_idx, 'ei': ei_parent}]

    C, T = snips_raw.shape[:2]
    p2p = ei_parent.ptp(axis=1)
    p2p_thresh = params.get('p2p_thresh_adc', 50)

    chan_sel = np.where(p2p >= p2p_thresh)[0]
    if chan_sel.size > 80:
        chan_sel = chan_sel[np.argsort(p2p[chan_sel])[-80:]]
    elif chan_sel.size < 10:
        chan_sel = np.argsort(p2p)[-10:]

    snips_sel = snips_raw[chan_sel, :, :]
    C_sel = snips_sel.shape[0]
    X = snips_sel.reshape(C_sel * T, N).T

    n_comp = min(7, X.shape[1] - 1)
    Xred = PCA(n_components=n_comp, svd_solver='randomized').fit_transform(X)

    labels = KMeans(k_split, n_init=5, random_state=depth).fit(Xred).labels_

    survivors = []
    for lab in np.unique(labels):
        mask = labels == lab
        if mask.sum() < min_spikes:
            continue

        snips_c = snips_raw[:, :, mask]
        ei_c = median_ei_adaptive(snips_c)
        pk_c = np.argmin(ei_c.min(axis=1))
        if pk_c != target_chan:
            continue

        gbm_c = compute_global_baseline_mean(ei_c)
        if gbm_c >= baseline_cut:
            continue

        print(f"            child gmb: {gbm_c}")

        survivors.append({'mask': mask,
                          'ei': ei_c,
                          'gbm': gbm_c})

    if not survivors:
        return [{'inds': abs_idx, 'ei': ei_parent}]

    merged = []
    taken = np.zeros(len(survivors), dtype=bool)

    for i, s_i in enumerate(survivors):
        if taken[i]:
            continue
        union_mask = s_i['mask'].copy()
        for j, s_j in enumerate(survivors[i + 1:], start=i + 1):
            if taken[j]:
                continue
            if ei_similarity(s_i['ei'], s_j['ei']) > ei_sim_thr:
                print(f"Merged some, sim {ei_similarity(s_i['ei'], s_j['ei']):0.2f}")
                union_mask |= s_j['mask']
                taken[j] = True

        snips_u = snips_raw[:, :, union_mask]
        merged.append({
            'mask': union_mask,
            'ei': median_ei_adaptive(snips_u),
            'abs': abs_idx[union_mask]
        })

    if len(merged) == 1:
        return [{'inds': merged[0]['abs'],
                 'ei': merged[0]['ei']}]

    out = []
    for child in merged:
        idxs = np.where(child['mask'])[0]
        st_sub = spike_times[idxs]
        sr_sub = snips_raw[:, :, child['mask']]
        out.extend(
            verify_clusters_recursive(st_sub,
                                      sr_sub,
                                      params,
                                      target_chan,
                                      abs_idx=child['abs'],
                                      depth=depth + 1)
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cross-correlation
# ─────────────────────────────────────────────────────────────────────────────

def xcorr_spike_times(st1, st2, max_lag=None, total_len=None):
    """
    MATLAB-style cross-correlation of two spike time lists (binary event vectors).

    Parameters
    ----------
    st1, st2 : array-like
        Spike times (sample indices).
    max_lag : int or None
        If given, restrict lags to [-max_lag, max_lag].
    total_len : int or None
        Total duration of recording (vector length).

    Returns
    -------
    lags : ndarray
    cc_norm : ndarray
        Normalized cross-correlation coefficients.
    """
    st1 = np.asarray(st1, dtype=int)
    st2 = np.asarray(st2, dtype=int)

    if total_len is None:
        total_len = max(st1.max() if st1.size > 0 else 0,
                        st2.max() if st2.size > 0 else 0) + 1

    vec1 = np.zeros(total_len, dtype=np.float32)
    vec2 = np.zeros(total_len, dtype=np.float32)
    vec1[st1] = 1
    vec2[st2] = 1

    cc_full = correlate(vec2, vec1, mode='full')
    lags = correlation_lags(len(vec2), len(vec1), mode='full')

    norm = np.sqrt(np.sum(vec1 ** 2) * np.sum(vec2 ** 2))
    cc_norm = cc_full / norm if norm > 0 else cc_full * 0

    if max_lag is not None:
        keep = np.abs(lags) <= max_lag
        lags = lags[keep]
        cc_norm = cc_norm[keep]

    return lags, cc_norm


# ─────────────────────────────────────────────────────────────────────────────
# Harm-map pipeline
# ─────────────────────────────────────────────────────────────────────────────

def select_template_channels(
    ei, p2p_thr=50.0, max_n=80, min_n=10, force_include_main=True
):
    """
    Return (channels, p2p) where channels are sorted by descending p2p.
    Ensures at least min_n channels by adding sub-threshold channels with highest p2p.
    """
    ptp = ei.max(axis=1) - ei.min(axis=1)
    C = ptp.size

    strong = np.flatnonzero(ptp >= p2p_thr)
    strong = strong[np.argsort(ptp[strong])[::-1]]

    if force_include_main:
        ch_main = int(np.argmin(ei.min(axis=1)))
    else:
        ch_main = None

    picked = list(strong)
    seen = set(picked)
    if ch_main is not None and ch_main not in seen:
        picked.append(ch_main); seen.add(ch_main)

    order_all = np.argsort(ptp)[::-1]
    for ch in order_all:
        if len(picked) >= min_n:
            break
        if ch not in seen:
            picked.append(ch); seen.add(ch)

    picked = sorted(picked, key=lambda c: ptp[c], reverse=True)[:max_n]

    if len(picked) == 0:
        picked = order_all[:min(min_n, C)].tolist()

    return np.asarray(picked, dtype=int), ptp


def main_channel_and_neg_peak(ei):
    """Main channel = most negative trough; return (channel_index, t_neg)."""
    mins = ei.min(axis=1)
    ch_main = int(np.argmin(mins))
    t_neg = int(np.argmin(ei[ch_main]))
    return ch_main, t_neg


def roll_zero_2d(arr, shift):
    """
    Zero-padded shift along time axis for 2D [nch, T].
    Positive shift -> moves waveform to the right.
    """
    nch, T = arr.shape
    out = np.zeros_like(arr)
    if shift == 0:
        out[:] = arr
    elif shift > 0:
        out[:, shift:] = arr[:, :T - shift]
    else:
        s = -shift
        out[:, :T - s] = arr[:, s:]
    return out


def compute_harm_map_noamp(
    ei,
    snips,
    p2p_thr=50.0,
    max_channels=80,
    min_channels=10,
    lag_radius=3,
    weight_by_p2p=True,
    weight_beta=0.5,
    force_include_main=True,
):
    C, T = ei.shape
    assert snips.shape[:2] == (C, T), "snips must be [C,T,N]"
    N = snips.shape[2]

    chans, ptp = select_template_channels(ei, p2p_thr, max_channels, min_channels)
    if force_include_main:
        ch_main, t_neg = main_channel_and_neg_peak(ei)
        if ch_main not in chans:
            chans = np.concatenate([chans[:-1], [ch_main]])
            chans = np.array(sorted(set(chans), key=lambda c: ptp[c], reverse=True), dtype=int)
    else:
        ch_main, t_neg = main_channel_and_neg_peak(ei)

    ei_sel = ei[chans]
    raw_sel = snips[chans]
    nch = ei_sel.shape[0]

    lags = np.arange(-lag_radius, lag_radius + 1, dtype=int)
    L = lags.size

    rms_raw = np.sqrt((raw_sel ** 2).mean(axis=1))

    if weight_by_p2p:
        w = (ptp[chans] ** weight_beta).astype(np.float32)
    else:
        w = np.ones(nch, dtype=np.float32)
    w /= w.sum()

    deltas = np.empty((L, nch, N), dtype=np.float32)
    for i, d in enumerate(lags):
        shifted = roll_zero_2d(ei_sel, d)
        resid = raw_sel - shifted[:, :, None]
        rms_res = np.sqrt((resid ** 2).mean(axis=1))
        deltas[i] = rms_res - rms_raw

    mean_deltas = (deltas * w[:, None]).sum(axis=1)
    best_i = np.argmin(mean_deltas, axis=0)
    best_lags = lags[best_i]

    if deltas.shape == (L, N, nch):
        deltas = deltas.transpose(0, 2, 1)
    elif deltas.shape != (L, nch, N):
        raise AssertionError(f"Unexpected deltas shape: {deltas.shape}")

    harm = np.take_along_axis(deltas, best_i[None, None, :], axis=0).squeeze(0)

    mean_delta_unweighted = harm.mean(axis=0)
    mean_delta_weighted = (harm * w[:, None]).sum(axis=0)
    p2p_sel = ptp[chans]

    out = {
        "selected_channels": chans,
        "channel_ptp": p2p_sel,
        "main_channel": ch_main,
        "neg_peak_index": t_neg,
        "lags": lags,
        "best_lag_per_spike": best_lags,
        "harm_matrix": harm,
        "mean_delta_unweighted": mean_delta_unweighted,
        "mean_delta_weighted": mean_delta_weighted,
    }
    return out


def compute_spike_gate(
    res,
    *,
    thr_global=-2.0,
    thr_channel=0.0,
    min_good_frac=0.5,
    max_bad_delta=5.0,
    weighted=True,
    weight_beta=0.5,
    ideal=None,
    exceed_thresh=None
):
    """
    Returns:
      accept_mask, global_mean, n_good, n_bad, frac_good,
      max_harm_delta, n_exceed_max_gap
    """
    H = res["harm_matrix"]
    ptp = res["channel_ptp"]
    nch, N = H.shape

    good = (H < thr_channel)
    harm = (H > thr_channel)

    n_good = good.sum(axis=0)
    n_bad = harm.sum(axis=0)
    frac_good = n_good / max(1, nch)

    if weighted:
        w = (ptp.astype(float) ** weight_beta)
    else:
        w = np.ones_like(ptp, float)
    w /= w.sum()

    global_mean = (H * w[:, None]).sum(axis=0)

    max_harm_delta = np.where(harm, H, -np.inf).max(axis=0)
    harm_cap_ok = (max_harm_delta <= max_bad_delta) | ~np.isfinite(max_harm_delta)

    if (ideal is not None) and (exceed_thresh is not None):
        mxt = np.asarray(ideal["max_trusted"], dtype=float)
        if mxt.shape[0] != nch:
            raise ValueError(f"ideal['max_trusted'] has nch={mxt.shape[0]} but res has nch={nch}")
        delta_minus_max = H - mxt[:, None]
        exceed_mat = delta_minus_max > float(exceed_thresh)
        n_exceed_max_gap = exceed_mat.sum(axis=0).astype(int)
        exceed_ok = (n_exceed_max_gap == 0)
    else:
        n_exceed_max_gap = np.zeros(N, dtype=int)
        exceed_ok = np.ones(N, dtype=bool)

    accept = (
        (global_mean < thr_global) &
        (frac_good >= min_good_frac) &
        harm_cap_ok &
        exceed_ok
    )

    return {
        "accept_mask": accept,
        "global_mean": global_mean,
        "n_good": n_good, "n_bad": n_bad,
        "frac_good": frac_good,
        "max_harm_delta": max_harm_delta,
        "n_exceed_max_gap": n_exceed_max_gap,
    }


def build_delta_prototype(
    res, *,
    subset=None,
    good_thresh=-5.0,
    top_frac=0.25,
    core_k=None,
    core_frac=0.5,
    min_periph=5,
    beta=0.7
):
    """
    Learn per-channel expected Δ profile (mu_c) and robust scale (mad_c) from very-good spikes.
    Returns dict with mu_c, mad_c, core_idx, periph_idx, weights, ptp, idx_good (global).
    """
    H = res["harm_matrix"]
    ptp = res["channel_ptp"]
    mdw = res["mean_delta_weighted"]
    nch, N = H.shape

    if subset is not None:
        subset = np.asarray(subset, dtype=int)
        pool_indices = subset
        mdw_pool = mdw[subset]
    else:
        pool_indices = np.arange(N, dtype=int)
        mdw_pool = mdw

    good_pool_local = np.where(mdw_pool <= good_thresh)[0]
    if good_pool_local.size == 0:
        order = np.argsort(mdw_pool)
        q = max(1, int(np.floor(order.size * top_frac)))
        good_local = order[:q]
    else:
        q = max(1, int(np.floor(good_pool_local.size * top_frac)))
        good_local = good_pool_local[np.argsort(mdw_pool[good_pool_local])[:q]]

    idx_good_global = pool_indices[good_local]

    G = H[:, idx_good_global]
    mu_c = np.median(G, axis=1)
    mad_c = np.median(np.abs(G - mu_c[:, None]), axis=1)
    mad_c = np.where(mad_c > 1e-6, mad_c, 1.0)

    if core_k is None:
        k = int(round(nch * core_frac))
    else:
        k = int(core_k)
    k = max(1, min(k, max(1, nch - min_periph)))

    order_by_ptp = np.argsort(ptp)[::-1]
    core_idx = order_by_ptp[:k]
    periph_idx = order_by_ptp[k:]

    w = (ptp.astype(float) ** beta)
    w_sum = w.sum()
    w = w / w_sum if w_sum > 0 else np.full_like(w, 1.0 / max(1, w.size), dtype=float)

    return {
        "mu_c": mu_c, "mad_c": mad_c,
        "core_idx": core_idx, "periph_idx": periph_idx,
        "weights": w, "ptp": ptp,
        "idx_good": idx_good_global,
    }


def build_ideal_delta(res, *, subset=None, good_thresh=-5.0, top_frac=0.25):
    """
    Learn per-channel 'ideal' Δ profile from a trusted subset of spikes.
    Returns per-channel: mu_c, var_c, max_trusted, trusted_idx.
    """
    H = res["harm_matrix"]
    mdw = res["mean_delta_weighted"]
    nch, N = H.shape

    pool = np.asarray(subset, dtype=int) if subset is not None else np.arange(N, dtype=int)
    mdw_pool = mdw[pool]

    elig_local = np.where(mdw_pool <= good_thresh)[0]
    if elig_local.size == 0:
        order = np.argsort(mdw_pool)
        q = max(1, int(np.floor(order.size * top_frac)))
        good_local = order[:q]
    else:
        q = max(1, int(np.floor(elig_local.size * top_frac)))
        good_local = elig_local[np.argsort(mdw_pool[elig_local])[:q]]

    trusted = pool[good_local]
    G = H[:, trusted]

    mu_c = np.median(G, axis=1)
    var_c = np.var(G, axis=1)
    max_trusted = np.max(G, axis=1)

    return {
        "mu_c": mu_c.astype(float),
        "var_c": var_c.astype(float),
        "max_trusted": max_trusted.astype(float),
        "trusted_idx": trusted
    }


def score_against_prototype(res, proto, *, harm_cap=10.0, tau_core=1.0, tau_periph=1.5):
    """
    For each spike, compute S_core/S_periph from z-shortfalls vs the prototype,
    plus max harmful Δ for your cap. Labels: {'good','near_miss','bad'}.
    """
    H = res["harm_matrix"]
    mu, mad = proto["mu_c"], proto["mad_c"]
    core, periph = proto["core_idx"], proto["periph_idx"]

    shortfall = np.maximum(0.0, H - mu[:, None])
    zshort = shortfall / mad[:, None]

    def safe_mean(arr, axis=0):
        return np.nanmean(np.where(np.isfinite(arr), arr, np.nan), axis=axis)

    S_core = safe_mean(zshort[core, :], axis=0)
    S_periph = safe_mean(zshort[periph, :], axis=0) if periph.size > 0 else np.zeros(H.shape[1])

    harm_mask = (H > 0)
    max_harm = np.where(harm_mask, H, -np.inf).max(axis=0)
    cap_ok = (max_harm <= harm_cap) | ~np.isfinite(max_harm)

    labels = np.empty(H.shape[1], dtype=object); labels[:] = "bad"
    good_core = (S_core <= tau_core) & cap_ok
    near_miss = good_core & (S_periph > tau_periph)
    good_core = good_core & ~near_miss
    labels[good_core] = "good"; labels[near_miss] = "near_miss"

    return {
        "S_core": S_core, "S_periph": S_periph,
        "max_harm": max_harm, "labels": labels,
        "good_mask": good_core, "near_miss_mask": near_miss
    }


def compute_profile_deviation_metrics(res, ideal, *, var_threshold=3.0, exceed_thresh=25.0):
    """
    From res and per-channel 'ideal' stats, compute per-spike diagnostics.
    Returns dict of 1D arrays length N.
    """
    H = res["harm_matrix"]
    mu = ideal["mu_c"]
    var = ideal["var_c"]
    mxt = ideal["max_trusted"]
    nch, N = H.shape

    thr = mu[:, None] + (var_threshold * var)[:, None]
    n_gt_mu_plus_varthr = (H > thr).sum(axis=0).astype(int)

    n_gt_half_max = (H > (0.5 * mxt)[:, None]).sum(axis=0).astype(int)

    trusted_help = (mxt < 0.0)[:, None]
    n_pos_when_trusted_lt0 = (trusted_help & (H > 0.0)).sum(axis=0).astype(int)

    delta_minus_max = H - mxt[:, None]
    max_delta_minus_max = delta_minus_max.max(axis=0)

    pos = (H > 0.0)
    sum_pos = (H * pos).sum(axis=0)
    cnt_pos = pos.sum(axis=0)
    mean_pos_delta = np.divide(sum_pos, np.maximum(cnt_pos, 1), where=(cnt_pos > 0))
    sum_mxt_on_pos = (mxt[:, None] * pos).sum(axis=0)
    mean_mxt_on_pos = np.divide(sum_mxt_on_pos, np.maximum(cnt_pos, 1), where=(cnt_pos > 0))
    mean_pos_delta_gap = np.where(cnt_pos > 0, (mean_pos_delta - mean_mxt_on_pos), 0.0)

    n_exceed_max_gap = (delta_minus_max > float(exceed_thresh)).sum(axis=0).astype(int)

    return dict(
        n_gt_mu_plus_varthr=n_gt_mu_plus_varthr,
        n_gt_half_max=n_gt_half_max,
        n_pos_when_trusted_lt0=n_pos_when_trusted_lt0,
        max_delta_minus_max=max_delta_minus_max,
        mean_pos_delta_gap=mean_pos_delta_gap,
        n_exceed_max_gap=n_exceed_max_gap,
    )


def compute_deviation_signals(res, mu_c, *, top_k=5):
    """
    dev_matrix = actualΔ − idealΔ (positive = worse than ideal).
    Returns per-spike: dev_mean_all, dev_mean_topk_bad, pos_counts.
    """
    H = res["harm_matrix"]
    D = H - mu_c[:, None]
    dev_mean_all = D.mean(axis=0)

    Dpos = np.maximum(0.0, D)
    Dpos_sorted = np.sort(Dpos, axis=0)[::-1, :]
    k = min(top_k, Dpos_sorted.shape[0])
    topk = Dpos_sorted[:k, :]
    pos_counts = (Dpos > 0).sum(axis=0)
    denom = np.clip(np.minimum(pos_counts, k), 1, None).astype(float)
    dev_mean_topk_bad = topk.sum(axis=0) / denom

    return {
        "dev_matrix": D,
        "dev_mean_all": dev_mean_all,
        "dev_mean_topk_bad": dev_mean_topk_bad,
        "pos_counts": pos_counts
    }


def _sanitize_order(order, N):
    if order is None:
        return np.arange(N, dtype=int)
    o = np.asarray(order, dtype=int).ravel()
    o = o[(o >= 0) & (o < N)]
    seen = set(); safe = []
    for i in o:
        if i not in seen:
            seen.add(i); safe.append(i)
    return np.asarray(safe if len(safe) > 0 else np.arange(N, dtype=int))


def _help_harm_by_spike(res, thr=0.0, spike_order=None, weighted=True, weight_beta=0.5):
    """
    Returns per-spike counts and (weighted) averages for 'help' (Δ<thr) and 'harm' (Δ>thr).
    """
    H = res["harm_matrix"]
    ptp = res["channel_ptp"]
    nch, N = H.shape

    if spike_order is None:
        order = np.arange(N)
    else:
        order = np.asarray(spike_order, dtype=int)
    H = H[:, order]

    if weighted:
        w = (ptp.astype(float) ** weight_beta)
    else:
        w = np.ones_like(ptp, dtype=float)

    help_mask = (H < thr)
    harm_mask = (H > thr)

    n_help = help_mask.sum(axis=0)
    n_harm = harm_mask.sum(axis=0)

    w_col_help = w[:, None] * help_mask
    w_col_harm = w[:, None] * harm_mask
    denom_help = w_col_help.sum(axis=0)
    denom_harm = w_col_harm.sum(axis=0)
    mean_help = (H * w[:, None] * help_mask).sum(axis=0) / np.where(denom_help > 0, denom_help, np.nan)
    mean_harm = (H * w[:, None] * harm_mask).sum(axis=0) / np.where(denom_harm > 0, denom_harm, np.nan)

    return {
        "order": order,
        "n_help": n_help, "n_harm": n_harm,
        "mean_help": mean_help, "mean_harm": mean_harm
    }


def _wilson_lower_bound(k, n, z=1.96):
    """Wilson score lower bound for a proportion, good for small N."""
    if n <= 0:
        return np.nan
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = phat + (z * z) / (2 * n)
    spread = z * np.sqrt((phat * (1.0 - phat) + (z * z) / (4 * n)) / n)
    return (center - spread) / denom


def lag_metrics_from_res(res, mask=None, central_band=1, z=1.96):
    """
    Compute lag metrics from compute_harm_map_noamp output.
      central_LB: Wilson lower bound of P(|lag|<=central_band)
      edge_frac : fraction at min/max lag of the bank
      mad       : median absolute deviation (around median lag)
    """
    best = np.asarray(res["best_lag_per_spike"])
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        best = best[mask]
    N = best.size
    if N == 0:
        return {"N": 0, "central_LB": np.nan, "edge_frac": np.nan, "mad": np.nan}

    lags = np.asarray(res["lags"])
    lmin, lmax = lags.min(), lags.max()

    central = np.abs(best) <= central_band
    central_LB = _wilson_lower_bound(int(central.sum()), N, z=z)

    edge_frac = np.mean((best == lmin) | (best == lmax))

    med = np.median(best)
    mad = np.median(np.abs(best - med))

    return {"N": int(N), "central_LB": float(central_LB), "edge_frac": float(edge_frac), "mad": float(mad)}


def reassign_leftovers_to_existing_templates(
    accepted_eis, snips_pool, spike_times_pool, pool_ids, SHOW_DIAGNOSTICS, cfg: SimpleNamespace
):
    """
    Try to place leftover spikes onto already-accepted templates.
    Uses the same harm/gate as elsewhere. Does NOT recalc EIs.
    Returns updated (snips_pool, spike_times_pool, pool_ids, placed_total).
    """
    placed_total = 0
    if (snips_pool.shape[2] == 0) or (len(accepted_eis) == 0):
        return snips_pool, spike_times_pool, pool_ids, placed_total

    while True:
        N = snips_pool.shape[2]
        if N == 0:
            break
        avail = np.ones(N, dtype=bool)
        placed_this_pass = 0

        for ti, tpl in enumerate(accepted_eis):
            ei_t = tpl['ei']
            res = compute_harm_map_noamp(
                ei_t, snips_pool,
                p2p_thr=cfg.HARM_P2P_THR, max_channels=cfg.HARM_MAX_CHANNELS,
                min_channels=cfg.HARM_MIN_CHANNELS,
                lag_radius=cfg.HARM_LAG_RADIUS, weight_by_p2p=cfg.WEIGHT_BY_P2P,
                weight_beta=cfg.WEIGHT_BETA
            )
            gate = compute_spike_gate(
                res,
                thr_global=cfg.GATE_THR_GLOBAL, thr_channel=0.0,
                min_good_frac=cfg.GATE_MIN_GOOD_FRAC, max_bad_delta=cfg.GATE_MAX_BAD_DELTA,
                weighted=cfg.WEIGHT_BY_P2P, weight_beta=cfg.WEIGHT_BETA
            )
            acc = (gate["accept_mask"] & avail)
            n_acc = int(acc.sum())
            if n_acc > 0:
                accepted_eis[ti]['spike_times'] = np.concatenate(
                    [accepted_eis[ti]['spike_times'], spike_times_pool[acc]]
                )
                avail[acc] = False
                placed_this_pass += n_acc
                if SHOW_DIAGNOSTICS:
                    print(f"Reassigned {n_acc} leftover spikes to template {ti}.")

        if placed_this_pass == 0:
            break

        snips_pool = snips_pool[:, :, avail]
        spike_times_pool = spike_times_pool[avail]
        pool_ids = pool_ids[avail]
        placed_total += placed_this_pass

    if SHOW_DIAGNOSTICS:
        print(f"Reassignment placed {placed_total} spikes into existing templates.")
    return snips_pool, spike_times_pool, pool_ids, placed_total


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_harm_heatmap(result, sort_by_ptp=True, spike_order=None, vclip=None, title=None,
                      vline_at=None, vline_kwargs=None):
    H = result["harm_matrix"]
    ptp = result["channel_ptp"]

    chan_order = np.argsort(-ptp) if sort_by_ptp else np.arange(H.shape[0])
    if spike_order is None:
        spike_order = np.arange(H.shape[1])
    else:
        spike_order = np.asarray(spike_order, dtype=int)

    Hs = H[chan_order][:, spike_order]

    if vclip is None:
        vmax = np.percentile(np.abs(Hs), 98)
    else:
        vmax = float(vclip)

    plt.figure(figsize=(20, 4))
    im = plt.imshow(Hs, aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label="ΔRMS (res − raw)  [neg = help]")
    plt.xlabel("Spike index (ordered)")
    plt.ylabel("Channels (sorted by EI p2p)" if sort_by_ptp else "Channels")
    if title:
        plt.title(title)

    if vline_at is not None:
        ax = plt.gca()
        opts = dict(color='k', linestyle='--', linewidth=1.5, alpha=0.9)
        if vline_kwargs:
            opts.update(vline_kwargs)

        def _to_xpos(x):
            if isinstance(x, (int, np.integer)):
                return float(x) - 0.5
            return float(x)

        if isinstance(vline_at, (list, tuple, np.ndarray)):
            for x in vline_at:
                ax.axvline(_to_xpos(x), **opts)
        else:
            ax.axvline(_to_xpos(vline_at), **opts)

    plt.tight_layout()


def plot_spike_delta_summary(result, weighted=True, bins=60, title=None):
    d = result["mean_delta_weighted"] if weighted else result["mean_delta_unweighted"]
    plt.figure(figsize=(6, 2))
    plt.hist(d, bins=bins)
    plt.axvline(-2, linestyle='--')
    plt.xlabel("Mean ΔRMS per spike")
    plt.ylabel("Count")
    if title:
        plt.title(title)
    plt.tight_layout()


def plot_help_harm_lines(res, thr=0.0, spike_order=None, weighted=True, weight_beta=0.5, title=None):
    m = _help_harm_by_spike(res, thr, spike_order, weighted, weight_beta)
    x = np.arange(m["n_help"].size)

    fig, ax = plt.subplots(1, 2, figsize=(12, 2), sharex=True)
    ax[0].plot(x, m["n_help"], label=f"help (Δ<{thr:g})")
    ax[0].plot(x, m["n_harm"], label=f"harm (Δ>{thr:g})")
    ax[0].set_ylabel("# channels")
    ax[0].set_xlabel("Spike index (ordered)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.25)

    ax[1].plot(x, m["mean_help"], label="mean Δ (help)")
    ax[1].plot(x, m["mean_harm"], label="mean Δ (harm)")
    ax[1].axhline(0, ls="--", lw=1)
    ax[1].set_ylabel("ΔRMS (res − raw)")
    ax[1].set_xlabel("Spike index (ordered)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.25)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return m


def plot_help_harm_scatter_swapped(
    res, thr=0.0, spike_order=None, weighted=True, weight_beta=0.5,
    title="Scatter: mean Δ vs #channels (color = |opposite mean Δ|)",
    cmap="YlGn_r", big_mask=None, big_thresh=None, s_small=14, s_big=48
):
    m = _help_harm_by_spike(res, thr, spike_order, weighted, weight_beta)
    order = m["order"]
    N = order.size

    global_mean = np.asarray(res["mean_delta_weighted"])[order]

    if big_mask is not None:
        big_mask = np.asarray(big_mask, dtype=bool)[order]
        sizes = np.where(big_mask, s_big, s_small).astype(float)
    elif big_thresh is not None:
        sizes = np.where(global_mean < big_thresh, s_big, s_small).astype(float)
    else:
        sizes = np.full(N, s_small, float)

    n_help, n_harm = m["n_help"], m["n_harm"]
    mean_help, mean_harm = m["mean_help"], m["mean_harm"]

    def norm_colors(vals):
        v = np.abs(vals).copy()
        if np.all(np.isnan(v)): v[:] = 0.0
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        if not np.isfinite(vmax - vmin) or (vmax - vmin) == 0: vmax = vmin + 1.0
        return cm.get_cmap(cmap)(mcolors.Normalize(vmin=vmin, vmax=vmax)(v))

    colors_help = norm_colors(mean_harm)
    colors_harm = norm_colors(mean_help)

    plt.figure(figsize=(6, 3))
    mh = ~np.isnan(mean_help)
    mb = ~np.isnan(mean_harm)

    plt.scatter(mean_help[mh], n_help[mh], c=colors_help[mh], s=sizes[mh],
                alpha=0.85, label=f"help (Δ<{thr:g})")
    plt.scatter(mean_harm[mb], n_harm[mb], c=colors_harm[mb], s=sizes[mb],
                alpha=0.85, label=f"harm (Δ>{thr:g})")

    plt.axvline(0, ls="--", lw=1)
    plt.xlabel("Group mean ΔRMS (res − raw)")
    plt.ylabel("# channels in group")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    return {
        "order": order,
        "sizes": sizes,
        "global_mean": global_mean,
        "n_help": n_help, "n_harm": n_harm,
        "mean_help": mean_help, "mean_harm": mean_harm
    }


def plot_ei_quick(ei, ei_positions, title, channel_of_interest, scale=90):
    plt.figure(figsize=(18, 8))
    plot_ei_waveforms(
        ei, ei_positions,
        ref_channel=channel_of_interest,
        scale=scale, box_height=1, box_width=50,
        colors='black', aspect=0.5)
    plt.title(title)
    plt.show()


def plot_deviation_lines(res, ideal, *, var_threshold=3.0, exceed_thresh=25.0,
                         spike_order=None, title=None):
    """
    Panels:
      [0] counts per spike
      [1] gap magnitudes
      [2] exceed count
    """
    N = res["harm_matrix"].shape[1]
    order = _sanitize_order(spike_order, N)

    metrics = compute_profile_deviation_metrics(
        res, ideal, var_threshold=var_threshold, exceed_thresh=exceed_thresh
    )

    x = np.arange(order.size)
    fig, ax = plt.subplots(1, 3, figsize=(20, 2), sharex=True)

    ax[0].plot(x, metrics["n_gt_mu_plus_varthr"][order], label=f"#(Δ > μ + {var_threshold}·var)")
    ax[0].plot(x, metrics["n_gt_half_max"][order], label="#(Δ > 0.5·max_trusted)")
    ax[0].plot(x, metrics["n_pos_when_trusted_lt0"][order], label="#(Δ>0 where max_trusted<0)")
    ax[0].set_ylabel("# channels"); ax[0].set_xlabel("Spike index"); ax[0].grid(True, alpha=0.25)
    ax[0].legend(loc="upper right", fontsize=8)

    ax[1].plot(x, metrics["max_delta_minus_max"][order], label="max(Δ − max_trusted)")
    ax[1].plot(x, metrics["mean_pos_delta_gap"][order], label="mean(Δ − max_trusted) on Δ>0")
    ax[1].axhline(0, ls="--", lw=1)
    ax[1].set_ylabel("Δ gap"); ax[1].set_xlabel("Spike index"); ax[1].grid(True, alpha=0.25)
    ax[1].legend(loc="upper right", fontsize=8)

    ax[2].plot(x, metrics["n_exceed_max_gap"][order], label=f"#(Δ − max_trusted > {exceed_thresh:g})")
    ax[2].set_ylabel("# channels"); ax[2].set_xlabel("Spike index"); ax[2].grid(True, alpha=0.25)
    ax[2].legend(loc="upper right", fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    return metrics


def main_channel_traces(snips, ch, idx, title):
    traces = snips[ch, :, idx]
    plt.figure(figsize=(12, 2))
    for tr in traces:
        plt.plot(tr, color='red', alpha=0.25)
    plt.plot(np.median(traces, axis=0), color='blue', lw=2)
    plt.title(title)
    plt.grid(True); plt.show()


def format_lag_metrics(metrics, central_band=1):
    """
    Compact string for titles, e.g.:
    'Lag: MAD=0.8, central±1 LB=0.72, edge=5.3% (N=446)'
    """
    if metrics["N"] == 0:
        return "Lag: N=0"
    return (f"Lag: MAD={metrics['mad']:.2f}, "
            f"central±{central_band} LB={metrics['central_LB']:.2f}, "
            f"edge={100 * metrics['edge_frac']:.1f}% (N={metrics['N']})")