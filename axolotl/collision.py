"""collision.py
Utility helpers for axon-based spike-collision modelling.

The module has **two tiers of lag handling**:

1. **`quick_unit_filter()`** – a *fast* pass that uses cross-correlation on the
   **peak channel only** to align each template, computes ΔRMS on the full
   selected-channel set, and returns the subset of units that produce a
   negative (i.e. improving) ΔRMS below a user threshold (default 0).

2. **`scan_unit_lags()`** – the existing exhaustive per-unit lag scan (using
   `lag_delta_rms`) run only on the units accepted by the quick filter.

The downstream API (evaluate_local_group → resolve_snippet →
accumulate_unit_stats etc.) is unchanged.

---------------------------------------------------------------------
Public symbols
--------------
roll_zero, roll_zero_all, tempered_weights, quick_unit_filter,
build_channel_index, lag_delta_rms, scan_unit_lags, score_active_set,
beam_combo_search, prune_combo, evaluate_local_group, is_certain,
resolve_snippet, accumulate_unit_stats, accept_units, micro_align_units,
subtract_overlap_tail, per_channel_gmm_bimodality, MAX_W_UNITS
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture

from .plotting import plot_ei_waveforms


# ─────────────────────────────────────────────────────────────────────────────
# Basic helpers
# ─────────────────────────────────────────────────────────────────────────────

def roll_zero(arr: np.ndarray, lag: int) -> np.ndarray:
    """Shift 1-D array by *lag* samples with zero-padding (no wrap-around)."""
    out = np.zeros_like(arr)
    if lag > 0:
        out[lag:] = arr[:-lag]
    elif lag < 0:
        out[:lag] = arr[-lag:]
    else:
        out[:] = arr
    return out


def roll_zero_all(ei: np.ndarray, lag: int) -> np.ndarray:
    """Shift all channels in [C, T] EI by lag samples, with zero-padding."""
    out = np.zeros_like(ei)
    if lag > 0:
        out[:, lag:] = ei[:, :-lag]
    elif lag < 0:
        out[:, :lag] = ei[:, -lag:]
    else:
        out[:] = ei
    return out


def tempered_weights(p2p_vec: np.ndarray, chans: Iterable[int], *, beta: float = 0.5) -> np.ndarray:
    """Normalised weights *w_c ∝ (p2p_c)^β* over **chans**."""
    w = p2p_vec[list(chans)] ** beta
    s = w.sum()
    return w / s if s else w


# ─────────────────────────────────────────────────────────────────────────────
# 0. Fast peak-channel filter
# ─────────────────────────────────────────────────────────────────────────────

def quick_unit_filter(
    unit_ids,
    raw_snippet: np.ndarray,
    unit_info: dict,
    *,
    delta_thr: float = 0.0,
    rms_raw
):
    """Fast x-corr gate; returns DataFrame of units that improve ΔRMS."""
    rows = []

    for uid in unit_ids:
        ei = unit_info[uid]['ei']
        peak_ch = unit_info[uid]['peak_channel']
        sel_ch = unit_info[uid]['selected_channels']

        if len(sel_ch) == 0:
            continue

        trace_rw = raw_snippet[sel_ch]
        trace_ei = ei[sel_ch]

        weights = trace_ei.max(axis=1) - trace_ei.min(axis=1)
        weights[weights > 200] = 200
        scores = []
        for lag in range(41):
            shifted_ei = roll_zero_all(trace_ei, lag)
            rms_res = np.sqrt(((trace_rw - shifted_ei) ** 2).mean(axis=1))
            delta = np.sum(weights * (rms_res - rms_raw[sel_ch]))
            scores.append((lag, delta))

        lag0, delta = min(scores, key=lambda x: x[1])

        if delta < delta_thr:
            rows.append({
                'uid': uid,
                'lag': lag0,
                'delta': delta,
                'peak_ch': peak_ch,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Channel-to-unit index
# ─────────────────────────────────────────────────────────────────────────────

def build_channel_index(good_units, unit_info):
    """
    good_units :  list / array / pandas-Series of uid strings  OR
                  DataFrame with a 'uid' column
    unit_info  : {uid: {'selected_channels': [...]}}
    Returns    : {channel: [uids]}
    """
    if hasattr(good_units, "columns") and "uid" in good_units.columns:
        uid_iter = good_units["uid"]
    else:
        uid_iter = good_units

    ch_map = defaultdict(list)
    for uid in uid_iter:
        for ch in unit_info[uid]["selected_channels"]:
            ch_map[ch].append(uid)
    return dict(sorted(ch_map.items()))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Per-unit ΔRMS sweep
# ─────────────────────────────────────────────────────────────────────────────

def lag_delta_rms(
    uid: int,
    raw_snippet: np.ndarray,
    p2p_all: Dict[int, np.ndarray],
    unit_info: Dict[int, Dict],
    *,
    beta: float = 0.5,
    amp_thr: float = 25.0,
    mask_thr: float = 5.0,
    max_lag: int = 60,
    rms_raw
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scan lags so that the EI's absolute peak (index *peak_idx*) lands in
    snippet samples 40…80.

    Returns
    -------
    lags   : 1-D array of tested lags
    score  : ΔRMS for each lag (-inf where channels had no weight)
    """
    ei = unit_info[uid]["ei"]

    peak_idx = 40
    target_center = 60
    base = target_center - peak_idx

    lag_min = base - max_lag
    lag_max = base + max_lag
    lags = np.arange(lag_min, lag_max + 1)

    a_ch = p2p_all[uid]
    chans = [c for c in unit_info[uid]["selected_channels"]
             if a_ch[c] >= amp_thr]
    if not chans:
        raise ValueError("no channels above amp_thr")

    W = tempered_weights(a_ch, chans, beta=beta)
    score = np.zeros_like(lags, dtype=np.float32)

    for i, lag in enumerate(lags):
        shifted_ei = roll_zero_all(ei[chans], lag)

        raw_sel = raw_snippet[chans]
        weights = shifted_ei.max(axis=1) - shifted_ei.min(axis=1)
        weights[weights > 200] = 200

        rms_res = np.sqrt(((raw_sel - shifted_ei) ** 2).mean(axis=1))
        delta = np.sum(weights * (rms_res - rms_raw[chans]))

        score[i] = -delta
    return lags, score


# ─────────────────────────────────────────────────────────────────────────────
# 2. Scan top-k lags for a set of units
# ─────────────────────────────────────────────────────────────────────────────

def scan_unit_lags(
    unit_ids: Iterable[int],
    raw_snippet: np.ndarray,
    p2p_all: Dict[int, np.ndarray],
    unit_info: Dict[int, Dict],
    *,
    beta: float = 0.5,
    amp_thr: float = 25.0,
    mask_thr: float = 5.0,
    max_lag: int = 60,
    top_k: int = 3,
    rms_raw
) -> Dict[int, List[int]]:
    """
    Build lag_dict {uid: [top_k lags]} for the supplied *unit_ids*.
    Only lags whose EI peak maps to snippet samples 40-80 are kept.
    """
    lag_dict = {}

    for uid in unit_ids:
        try:
            lags, score = lag_delta_rms(
                uid, raw_snippet, p2p_all, unit_info,
                beta=beta, amp_thr=amp_thr,
                mask_thr=mask_thr, max_lag=max_lag, rms_raw=rms_raw
            )
            keep = np.isfinite(score)
            if not keep.any():
                continue

            lags = lags[keep]
            score = score[keep]
            order = np.argsort(score)[::-1][:top_k]
            lag_dict[uid] = [int(lags[j]) for j in order]

        except Exception:
            continue

    return lag_dict


# ─────────────────────────────────────────────────────────────────────────────
# Combo-scoring primitives
# ─────────────────────────────────────────────────────────────────────────────

def score_active_set(
    active_dict: Dict[int, int],
    union_chans: Sequence[int],
    raw_local: np.ndarray,
    unit_info: Dict[int, Dict],
    p2p_all: Dict[int, np.ndarray],
    rolled_bank,
    *,
    beta: float = 0.5,
    rms_raw,
    debug: bool = False,
    chan_weights=None,
) -> float:
    """Return weighted ΔRMS for the given unit set."""
    if not active_dict:
        return 0.0

    tmpl_sum = None
    for idx, (uid, lag) in enumerate(active_dict.items()):
        if idx == 0:
            tmpl_sum = rolled_bank[(uid, lag)].copy()
        else:
            tmpl_sum += rolled_bank[(uid, lag)]

    tmpl_sum = tmpl_sum[union_chans]

    if chan_weights is None:
        weights = np.zeros(len(union_chans), dtype=np.float32)
        for k, c in enumerate(union_chans):
            a_u = max(p2p_all[u][c] for u in active_dict)
            weights[k] = a_u ** beta
    else:
        weights = chan_weights

    rms_res = np.sqrt(((raw_local - tmpl_sum) ** 2).mean(axis=1))
    if debug:
        diffs = weights * (rms_res - rms_raw[union_chans])
        for ch, val in zip(union_chans, diffs):
            print(f"channel {ch}: {val:.1f}")
    delta = np.sum(weights * (rms_res - rms_raw[union_chans]))

    return -delta


# ─────────────────────────────────────────────────────────────────────────────
# Beam search
# ─────────────────────────────────────────────────────────────────────────────

def beam_combo_search(units, lag_dict, union_chans, raw_local,
                      unit_info, p2p_all, rolled_bank,
                      *, beta=0.5, beam=4, rms_raw):
    """
    Parameters
    ----------
    units        : list[int]
    lag_dict     : {uid: [lag1, lag2, …]}
    union_chans  : list[int]
    raw_local    : ndarray (len(union_chans), T)

    Returns
    -------
    best_combo   : dict {'lags': {uid:lag}, 'score': ΔRMS}
    """
    beams = [({}, 0.0)]

    for u in units:
        new_beams = []
        for active, sc in beams:
            new_beams.append((active, sc))

            for L in lag_dict[u]:
                active2 = dict(active)
                active2[u] = L
                s2 = score_active_set(
                    active2, union_chans,
                    raw_local, unit_info, p2p_all, rolled_bank,
                    beta=beta, rms_raw=rms_raw
                )
                new_beams.append((active2, s2))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam]

    best_active, best_score = beams[0]
    return {'lags': best_active, 'score': best_score}


def prune_combo(active, union_chans, raw_local, unit_info, p2p_all, rolled_bank, rms_raw,
                beta=0.5):
    changed = True
    while changed:
        changed = False
        score_full = score_active_set(active, union_chans, raw_local,
                                      unit_info, p2p_all, rolled_bank,
                                      beta=beta, rms_raw=rms_raw)
        worst_uid, worst_gain = None, None
        for u in list(active):
            s_minus = score_active_set({k: v for k, v in active.items() if k != u},
                                       union_chans, raw_local,
                                       unit_info, p2p_all, rolled_bank,
                                       beta=beta, rms_raw=rms_raw)
            gain = score_full - s_minus
            if worst_gain is None or gain < worst_gain:
                worst_uid, worst_gain = u, gain
        if worst_gain is not None and worst_gain <= 0:
            del active[worst_uid]
            changed = True
    return active


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate local group
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_local_group(
    c0: int,
    working_units: Sequence[int],
    raw_snippet: np.ndarray,
    unit_info: Dict[int, Dict],
    lag_dict: Dict[int, List[int]],
    p2p_all: Dict[int, np.ndarray],
    rolled_bank,
    *,
    amp_thr: float = 25.0,
    mask_thr: float = 5.0,
    beta: float = 0.5,
    beam: int = 4,
    rms_raw,
    plot: bool = False,
):
    """Pick best lag-combo for one anchor channel using beam search.

    Returns
    -------
    best_combo      : dict  {'lags': {uid: lag}, 'score': ΔRMS}
    per_unit_delta  : dict  {uid: marginal ΔRMS vs empty set}
    """
    union = {c0}
    for u in working_units:
        union.update(np.where(p2p_all[u] >= amp_thr)[0])
    union_chans = sorted(union)
    raw_local = raw_snippet[union_chans]

    best_combo = beam_combo_search(
        units=working_units,
        lag_dict=lag_dict,
        union_chans=union_chans,
        raw_local=raw_local,
        unit_info=unit_info,
        p2p_all=p2p_all,
        rolled_bank=rolled_bank,
        beta=beta,
        beam=beam,
        rms_raw=rms_raw
    )

    pruned_lags = prune_combo(
        dict(best_combo['lags']),
        union_chans, raw_local,
        unit_info, p2p_all, rolled_bank, rms_raw
    )
    best_combo['lags'] = pruned_lags
    best_combo['score'] = score_active_set(
        pruned_lags, union_chans,
        raw_local, unit_info, p2p_all, rolled_bank,
        beta=beta, rms_raw=rms_raw)

    full_score = best_combo['score'] if best_combo else 0.0

    per_unit_delta = {}
    for u in working_units:
        per_unit_delta[u] = full_score if u in best_combo.get('lags', {}) else 0.0

    return best_combo, per_unit_delta


MAX_W_UNITS = 15  # hard cap per anchor channel


def is_certain(uid, unit_log, pos_thresh=3, neg_thresh=3):
    """
    Return True if this uid was either:
      • placed with positive ΔRMS ≥ twice, OR
      • examined ≥ (pos+neg) times but never had ΔRMS > 0
    """
    rec = unit_log.get(uid)
    if not rec:
        return False
    pos = sum(d > 0 for d in rec["deltas"])
    neg = sum(d <= 0 for d in rec["deltas"])
    return (pos >= pos_thresh) or (neg >= neg_thresh and pos == 0)


# ─────────────────────────────────────────────────────────────────────────────
# Snippet-level resolver
# ─────────────────────────────────────────────────────────────────────────────

def resolve_snippet(
    raw_snippet: np.ndarray,
    good_units: Sequence[int],
    channel_to_units: Dict[int, List[int]],
    lag_dict: Dict[int, List[int]],
    unit_info: Dict[int, Dict],
    p2p_all: Dict[int, np.ndarray],
    *,
    amp_thr: float = 25.0,
    beta: float = 0.5,
    beam: int = 4,
    rms_raw,
    rolled_bank,
    ei_positions,
    plot: bool = False,
):
    """Run the unresolved-channel loop for one snippet.

    Returns
    -------
    best_combo      : dict  {"lags": {uid:lag}, "score": ΔRMS}
    per_unit_delta  : dict  {uid: cumulative ΔRMS across anchor iterations}
    combo_history   : list  full trace of best_combo per anchor iteration
    """
    raw_ptp = raw_snippet.ptp(axis=1)
    unresolved_chans = {ch for uid in good_units for ch in unit_info[uid]["selected_channels"]}

    unit_log = defaultdict(lambda: {"deltas": [], "lags": []})
    combo_history = []

    while unresolved_chans:
        c0 = max(unresolved_chans, key=lambda c: raw_ptp[c])

        W_full = channel_to_units[c0]

        W = [u for u in W_full if not is_certain(u, unit_log)]

        if len(W) > MAX_W_UNITS:
            W.sort(key=lambda u: p2p_all[u][c0], reverse=True)
            W = W[:MAX_W_UNITS]

        if not W:
            W = [max(W_full, key=lambda u: p2p_all[u][c0])]

        best_combo, per_unit_delta_anchor = evaluate_local_group(
            c0, W, raw_snippet, unit_info, lag_dict, p2p_all, rolled_bank,
            amp_thr=amp_thr, beta=beta, beam=beam, rms_raw=rms_raw, plot=plot
        )
        combo_history.append(best_combo)

        for uid in per_unit_delta_anchor:
            unit_log[uid]["deltas"].append(per_unit_delta_anchor[uid])
            unit_log[uid]["lags"].append(best_combo["lags"].get(uid, math.nan))

        W_set = set(W)
        for ch in list(unresolved_chans):
            remaining = [u for u in channel_to_units[ch] if not is_certain(u, unit_log)]
            if set(remaining).issubset(W_set):
                unresolved_chans.remove(ch)

    # Build initial global combo from accumulated logs
    unit_log = {u: rec for u, rec in unit_log.items()
                if not np.all(np.isnan(rec["lags"]))}

    active = {}
    for u, rec in unit_log.items():
        if np.all(np.isnan(rec["lags"])):
            continue
        lag = float(np.nanmedian(rec["lags"]))
        if np.isnan(lag):
            continue
        if u not in lag_dict:
            continue
        lag_rounded = int(round(lag))
        allowed_lags = lag_dict[u]
        if lag_rounded not in allowed_lags:
            lag_rounded = min(allowed_lags, key=lambda l: abs(l - lag_rounded))
        active[u] = lag_rounded

    union_chans = sorted({c for u in active
                          for c in unit_info[u]["selected_channels"]})
    raw_local = raw_snippet[union_chans]

    def _template_sum(active_dict, union_chans, rolled_bank):
        """Return [len(union_chans), T] sum of rolled templates in active_dict."""
        if not active_dict:
            return np.zeros_like(raw_local, dtype=np.float32)

        tmpl_sum = np.zeros_like(raw_local, dtype=np.float32)
        for uid, lag in active_dict.items():
            rolled = rolled_bank[(uid, lag)][union_chans]
            tmpl_sum += rolled
        return tmpl_sum

    def marginal_prune(active_dict, local_beta):
        while len(active_dict) > 1:
            worst_uid, worst_gain = None, 0.0

            for uid in list(active_dict):
                sel = unit_info[uid]['selected_channels']
                weights = np.zeros(len(union_chans), dtype=np.float32)
                for k, c in enumerate(union_chans):
                    if c in sel:
                        weights[k] = (p2p_all[uid][c]) ** local_beta

                w_sum = weights.sum()
                if w_sum > 0:
                    weights /= w_sum

                full = score_active_set(active_dict, union_chans, raw_local,
                                        unit_info, p2p_all, rolled_bank,
                                        beta=local_beta, rms_raw=rms_raw,
                                        chan_weights=weights)

                alt = dict(active_dict); alt.pop(uid)
                minus = score_active_set(alt, union_chans, raw_local,
                                         unit_info, p2p_all, rolled_bank,
                                         beta=local_beta, rms_raw=rms_raw,
                                         chan_weights=weights)

                gain = full - minus
                if gain <= worst_gain:
                    worst_uid, worst_gain = uid, gain

            if worst_uid is None or worst_gain > 0:
                break

            active_dict.pop(worst_uid)

        return active_dict

    pruned = []
    if len(active) > 0:
        pruned = marginal_prune(active, local_beta=0.5)

    # Per-unit marginal ΔRMS on pruned set
    per_unit_delta = {}
    if len(pruned) > 0:
        for uid in list(pruned):
            sel = set(unit_info[uid]["selected_channels"])
            weights = np.zeros(len(union_chans), dtype=np.float32)

            for k, c in enumerate(union_chans):
                if c in sel:
                    weights[k] = (p2p_all[uid][c]) ** beta

            score_full = score_active_set(
                pruned, union_chans, raw_local,
                unit_info, p2p_all, rolled_bank,
                beta=beta, rms_raw=rms_raw,
                chan_weights=weights
            )

            alt = dict(pruned)
            alt.pop(uid)

            score_minus = score_active_set(
                alt, union_chans, raw_local,
                unit_info, p2p_all, rolled_bank,
                beta=beta, rms_raw=rms_raw,
                chan_weights=weights
            )

            per_unit_delta[uid] = score_full - score_minus

    if len(pruned) > 0:
        best_combo_global = {"lags": pruned, "score": score_full}
    else:
        best_combo_global = {}

    return best_combo_global, per_unit_delta, combo_history


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance & tuning
# ─────────────────────────────────────────────────────────────────────────────

def _robust_median(x: Sequence[float]) -> float:
    return float(np.nanmedian(x)) if len(x) else np.nan


def _robust_mad(x: Sequence[float], med: float) -> float:
    x = np.asarray(x, float)
    return float(np.nanmedian(np.abs(x - med))) if len(x) else np.nan


def accumulate_unit_stats(unit_log: Dict[int, Dict]) -> Dict[int, Dict]:
    """Aggregate *unit_log* into per-unit statistics dictionary."""
    stats = {}
    for uid, rec in unit_log.items():
        d = np.asarray(rec["deltas"], float)
        L = np.asarray(rec["lags"], float)

        pos_mask = d > 0
        lag_mask = ~np.isnan(L) & pos_mask

        pos_delta = d[pos_mask]
        good_lags = L[lag_mask]

        stats[uid] = {
            "delta_sum": float(np.nansum(pos_delta)),
            "delta_pos": float(np.nansum(pos_delta)),
            "delta_neg": float(np.nansum(d[d < 0])),
            "count_pos": int(pos_mask.sum()),
            "count_neg": int((d <= 0).sum()),
            "lag_med": np.nanmedian(good_lags) if good_lags.size else np.nan,
            "lag_mad": np.nanmedian(np.abs(good_lags - np.nanmedian(good_lags)))
                       if good_lags.size else np.inf
        }
    return stats


def accept_units(
    stats: Dict[int, Dict],
    *,
    pos_min: float = 20.0,
    net_min: float = 10.0,
    h_max: float = 0.3,
    lag_mad_max: float = 2,
    lag_med_max: float = 38,
) -> Tuple[List[int], List[int]]:
    """Return (accepted_uids, rejected_uids) based on hard thresholds."""
    accepted = []
    for uid, s in stats.items():
        P = s["delta_pos"]
        N = s["delta_neg"]
        H = abs(N) / P if P else np.inf
        net = P + N
        if (P >= pos_min and net >= net_min and H <= h_max
                and s["lag_mad"] <= lag_mad_max and s["lag_med"] <= lag_med_max):
            accepted.append(uid)
    rejected = [u for u in stats if u not in accepted]
    return accepted, rejected


def micro_align_units(
    accepted: Sequence[int],
    stats: Dict[int, Dict],
    unit_info: Dict[int, Dict],
    raw_snippet: np.ndarray,
    p2p_all: Dict[int, np.ndarray],
    rms_raw,
    *,
    mask_thr: float = 5.0,
    beta: float = 0.5,
    micro_sweep: int = 2,
) -> Dict[int, int]:
    """Fine-tune lags around median using ±*micro_sweep* neighbourhood."""
    final_lags: Dict[int, int] = {}
    final_deltas: Dict[int, int] = {}
    for uid in accepted:
        best_lag = int(round(stats[uid]["lag_med"]))
        best_score = -np.inf
        for d in range(-micro_sweep, micro_sweep + 1):
            lag = best_lag + d
            sel_ch = unit_info[uid]["selected_channels"]

            ei_full = unit_info[uid]["ei"].astype(np.float32)
            rolled_bank = {
                (uid, lag): np.roll(ei_full, shift=lag, axis=1)
            }
            score = score_active_set({uid: lag}, sel_ch, raw_snippet[sel_ch],
                                     unit_info, p2p_all, rolled_bank,
                                     beta=beta, rms_raw=rms_raw)

            if score > best_score:
                best_score = score
                final_lags[uid] = lag

        final_deltas[uid] = best_score

    return final_lags, final_deltas


# ─────────────────────────────────────────────────────────────────────────────
# Overlap subtraction
# ─────────────────────────────────────────────────────────────────────────────

def subtract_overlap_tail(
    raw_next_snip: np.ndarray,
    accepted_prev: Dict[int, int],
    unit_info: Dict[int, Dict],
    p2p_all: Dict[int, np.ndarray],
    *,
    overlap: int = 20,
    abs_thr: float = 2.0,
) -> np.ndarray:
    """Subtract tails of templates that spill into the next snippet window."""
    C, T = raw_next_snip.shape
    for uid, lag_prev in accepted_prev.items():
        ei = unit_info[uid]["ei"]
        tmpl = roll_zero(ei, lag_prev)
        start = tmpl.shape[1] - overlap
        end = start + T
        if start >= tmpl.shape[1]:
            continue
        tmpl_slice = tmpl[:, max(0, start): min(end, tmpl.shape[1])]
        dst_start = max(0, -start)
        dst_end = dst_start + tmpl_slice.shape[1]
        chan_mask = p2p_all[uid] >= abs_thr
        raw_next_snip[chan_mask, dst_start:dst_end] -= tmpl_slice[chan_mask]
    return raw_next_snip


# ─────────────────────────────────────────────────────────────────────────────
# GMM bimodality (core pipeline version)
# ─────────────────────────────────────────────────────────────────────────────

def per_channel_gmm_bimodality(
    ei, snips, n_top=5, min_sep=2.0,
    include_ref40=True, win40=3, min_cluster_size=20,
    pool_ids=None
):
    """
    For each channel, evaluate 2-GMM separation at multiple candidate times.
    Returns top n_top channels by separation score.

    Keys in each output dict:
      sep, chan, t, vmin, vmax, thr, mu_lo, mu_hi, std_lo, std_hi,
      n_lo, n_hi, polarity, cand_mask, other_mask,
      cand_idx, other_idx, cand_idx_global, other_idx_global,
      amp_lo, amp_hi, score_amp
    """
    C, T, N = snips.shape
    out = []

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

            mu = gmm.means_.flatten()
            std = np.array([np.sqrt(gmm.covariances_[k][0, 0]) for k in (0, 1)], dtype=float)

            order = np.argsort(mu)
            mu_lo, mu_hi = mu[order[0]], mu[order[1]]
            std_lo, std_hi = std[order[0]], std[order[1]]
            lab_lo = (labels == order[0])
            lab_hi = (labels == order[1])

            sep = abs(mu_hi - mu_lo) / np.sqrt(0.5 * (std_lo**2 + std_hi**2))

            if sep >= min_sep and (best is None or sep > best[0]):
                vflat = v.ravel()
                best = (
                    float(sep), int(t),
                    float(vflat.min()), float(vflat.max()),
                    lab_lo.copy(), lab_hi.copy(),
                    float(mu_lo), float(mu_hi),
                    float(std_lo), float(std_hi)
                )

        if best is None:
            continue

        sep, t_best, vmin, vmax, lab_lo, lab_hi, mu_lo, mu_hi, std_lo, std_hi = best

        pol = float(np.sign(ei[c, t_best]))
        if pol < 0:
            cand_mask = lab_lo
            other_mask = lab_hi
        else:
            cand_mask = lab_hi
            other_mask = lab_lo

        n_lo = int(lab_lo.sum()); n_hi = int(lab_hi.sum())
        cand_idx = np.where(cand_mask)[0]
        other_idx = np.where(other_mask)[0]

        if pool_ids is not None:
            cand_idx_global = pool_ids[cand_idx]
            other_idx_global = pool_ids[other_idx]
        else:
            cand_idx_global = None
            other_idx_global = None

        wav_lo = np.median(snips[c, :, lab_lo], axis=1) if n_lo > 0 else np.zeros(T, dtype=np.float32)
        wav_hi = np.median(snips[c, :, lab_hi], axis=1) if n_hi > 0 else np.zeros(T, dtype=np.float32)
        amp_lo = float(wav_lo.max() - wav_lo.min())
        amp_hi = float(wav_hi.max() - wav_hi.min())
        score_amp = max(amp_lo, amp_hi)

        thr = 0.5 * (mu_lo + mu_hi)

        out.append(dict(
            sep=sep, chan=int(c), t=int(t_best),
            vmin=vmin, vmax=vmax,
            thr=float(thr),
            mu_lo=float(mu_lo), mu_hi=float(mu_hi),
            std_lo=float(std_lo), std_hi=float(std_hi),
            n_lo=n_lo, n_hi=n_hi,
            polarity=pol,
            cand_mask=cand_mask, other_mask=other_mask,
            cand_idx=cand_idx, other_idx=other_idx,
            cand_idx_global=cand_idx_global, other_idx_global=other_idx_global,
            amp_lo=amp_lo, amp_hi=amp_hi, score_amp=score_amp
        ))

    out.sort(key=lambda d: -d['sep'])
    return out[:n_top]