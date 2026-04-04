# Axolotl Cleanup Plan — Running Log & Planner

## Completed

### Step 1: Split `collision.py` into `collision.py` + `collision_diagnostics.py` ✅
- **`collision.py`**: 868 lines — core collision resolution pipeline
- **`collision_diagnostics.py`**: 1452 lines — diagnostic, analysis, and plotting helpers
- Functions moved to diagnostics: `delta_rms`, `best_shift_ei`, `_peak_channel`, `_best_lag`, `marginal_gain`, `per_channel_gmm_bimodality_simple`, `per_channel_hist_otsu_candidates`, `bimodality_probe`, `recursive_bimodal_split`, `compute_global_baseline_mean`, `ei_similarity`, `median_ei_adaptive`, `verify_clusters_recursive`, `xcorr_spike_times`, `select_template_channels`, `main_channel_and_neg_peak`, `roll_zero_2d`, `compute_harm_map_noamp`, `compute_spike_gate`, `build_delta_prototype`, `build_ideal_delta`, `score_against_prototype`, `compute_profile_deviation_metrics`, `compute_deviation_signals`, `_sanitize_order`, `_help_harm_by_spike`, `_wilson_lower_bound`, `lag_metrics_from_res`, `reassign_leftovers_to_existing_templates`, and all plotting functions (`plot_harm_heatmap`, `plot_spike_delta_summary`, `plot_help_harm_lines`, `plot_help_harm_scatter_swapped`, `plot_ei_quick`, `plot_deviation_lines`, `main_channel_traces`, `format_lag_metrics`)

### Step 2: Clean up imports ✅
- `collision.py`: clean top-level imports only (numpy, matplotlib, pandas, sklearn.mixture, defaultdict, math)
- `collision_diagnostics.py`: clean imports + imports `score_active_set` from `.collision`

### Step 3: Update `run_axolotl.py` imports ✅
- `median_ei_adaptive` now imported from `axolotl.collision_diagnostics`

---

## Open Issues (found during review)

### HIGH — `marginal_gain()` always returns `None` (collision_diagnostics.py ~line 127)
The function iterates over units, finds ones with positive gain, **prints** them, but then falls through and **always returns `None`** instead of returning `(uid, lag, gain)` as documented in its docstring.

**Fix:** Replace `print(uid, lag, gain)` + fallthrough with `return uid, lag, gain`.

### CRITICAL — `compute_harm_map_noamp()` calls `main_channel_and_neg_peak(...` (collision_diagnostics.py line 758)
The `...` is Python's Ellipsis literal, not a placeholder comment. At runtime this passes `Ellipsis` as the `ei` argument to `main_channel_and_neg_peak`, which will crash with an `AttributeError`.

**Fix:** Replace `main_channel_and_neg_peak(...` with `main_channel_and_neg_peak(ei)`.

### MEDIUM — `rms_raw` parameter has no type annotation in 7+ functions (collision.py)
Functions `quick_unit_filter`, `lag_delta_rms`, `scan_unit_lags`, `score_active_set`, `beam_combo_search`, `prune_combo`, `evaluate_local_group`, `micro_align_units` all have `rms_raw` as a bare keyword-only parameter with no type hint and no default. Every caller must provide it, but it's easy to forget.

**Fix:** Add `rms_raw: np.ndarray` type annotation to all occurrences.

### MEDIUM — `median_ei_adaptive` imported twice in `run_axolotl.py` (lines 26 and 29)
```python
from axolotl.waveform_utils import ..., median_ei_adaptive      # line 26
from axolotl.collision_diagnostics import median_ei_adaptive      # line 29 (shadows above)
```
The second import silently shadows the first. Both likely have identical implementations, but this is fragile.

**Fix:** Remove the `waveform_utils` import of `median_ei_adaptive` (keep only the `collision_diagnostics` one), or verify they're identical and add a comment.

### LOW — Duplicate imports inside `main()` in `run_axolotl.py` (lines ~95-100)
Same `axolotl.*` modules imported at top of file and again inside `main()`. The inner imports are redundant.

**Fix:** Remove the duplicate imports inside `main()`.

### LOW — `cfg.MIN_CLUSTER_SIZE` in `recursive_bimodal_split` (collision_diagnostics.py)
Uses `cfg.MIN_CLUSTER_SIZE` but other functions use `min_cluster_size` as a direct parameter. If `cfg` is a `SimpleNamespace` without this attribute, it crashes.

**Fix:** Document required `cfg` attributes, or fall back to a default: `getattr(cfg, 'MIN_CLUSTER_SIZE', 20)`.

### LOW — `scipy.sparse.csgraph` used in `run_axolotl.py` without availability check
`merge_duplicate_units` imports `scipy.sparse.csgraph.connected_components` at runtime. If scipy isn't installed, this crashes.

**Fix:** Add a try/except import guard or document scipy as a hard dependency.

---

## Future / Wishlist

- [ ] Add type hints to all `rms_raw` parameters
- [ ] Fix `marginal_gain()` to actually return `(uid, lag, gain)`
- [ ] Fix `main_channel_and_neg_peak(...)` Ellipsis bug
- [ ] Clean up duplicate imports in `run_axolotl.py`
- [ ] Add docstring to `compute_harm_map_noamp` (currently missing)
- [ ] Consider adding `__all__` exports to both files for clarity
- [ ] Add unit tests for core collision functions
- [ ] Consider making `rms_raw` optional with a computed default

---

## File Sizes (current)

| File | Lines |
|------|-------|
| `axolotl/collision.py` | 868 |
| `axolotl/collision_diagnostics.py` | 1452 |
| `run_axolotl.py` | 422 |
| **Total** | **2742** |
