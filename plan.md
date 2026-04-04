# Cleanup Plan for `axolotl/collision.py`

## Problem
`collision.py` is ~3017 lines, bloated with:
- ~500+ lines of commented-out code blocks
- Duplicate function definitions
- Duplicate/redundant imports
- Functions defined but never called anywhere in the codebase
- Scattered `import numpy as np`, `import matplotlib.pyplot as plt`, etc. repeated mid-file

## Strategy
Split into two files:
1. **`axolotl/collision.py`** — core collision resolution pipeline (kept lean)
2. **`axolotl/collision_diagnostics.py`** — diagnostic, analysis, and plotting helpers

---

## Step 1: Create `axolotl/collision_diagnostics.py`

Move the following functions (useful for diagnostics/analysis but not part of the core pipeline):

### Metric helpers
- `delta_rms(x, y, weights)` — ΔRMS metric helper
- `best_shift_ei(trace_rw, trace_ei, max_shift)` — alignment utility
- `_peak_channel(ei)` — peak channel finder
- `_best_lag(raw_chan, ei_chan, peak_sample, max_lag)` — dot-product lag search
- `marginal_gain(...)` — greedy unit addition scorer (depends on `score_active_set` from collision.py → will import it)

### Bimodality / clustering
- `per_channel_gmm_bimodality_simple(...)` — simplified GMM bimodality
- `per_channel_hist_otsu_candidates(...)` — Otsu-based candidate finder
- `bimodality_probe(...)` — bimodality probe at channel/time
- `recursive_bimodal_split(...)` — recursive bimodal cluster splitter

### Baseline / similarity
- `compute_global_baseline_mean(...)` — baseline quality metric
- `ei_similarity(ei_a, ei_b)` — cosine similarity of templates
- `median_ei_adaptive(...)` — robust EI computation (also used by `run_axolotl.py` — will re-export)

### Harm-map pipeline
- `select_template_channels(...)` — channel selection by p2p
- `main_channel_and_neg_peak(...)` — find main channel
- `roll_zero_2d(...)` — 2D zero-padded shift
- `compute_harm_map_noamp(...)` — full harm map computation
- `compute_spike_gate(...)` — spike acceptance gating
- `build_delta_prototype(...)` — learn expected Δ profile
- `build_ideal_delta(...)` — learn ideal Δ stats
- `score_against_prototype(...)` — score spikes vs prototype
- `compute_profile_deviation_metrics(...)` — per-spike deviation diagnostics
- `compute_deviation_signals(...)` — deviation signal extraction
- `_sanitize_order(...)` — helper for spike ordering
- `_help_harm_by_spike(...)` — per-spike help/harm counts
- `_wilson_lower_bound(...)` — Wilson score CI
- `lag_metrics_from_res(...)` — lag quality metrics
- `reassign_leftovers_to_existing_templates(...)` — leftover spike reassignment

### Plotting
- `plot_harm_heatmap(...)`
- `plot_spike_delta_summary(...)`
- `plot_help_harm_lines(...)`
- `plot_help_harm_scatter_swapped(...)`
- `plot_ei_quick(...)`
- `plot_deviation_lines(...)`
- `main_channel_traces(...)`
- `format_lag_metrics(...)`

### Cross-correlation
- `xcorr_spike_times(...)` — spike train cross-correlation

### Recursive cluster validation
- `verify_clusters_recursive(...)` — recursive cluster splitter (depends on `median_ei_adaptive`, `compute_global_baseline_mean`, `ei_similarity` — all moved to same file)

### Imports for diagnostics file
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.signal import correlate, correlation_lags
from types import SimpleNamespace
from .plotting import plot_ei_waveforms
from .collision import score_active_set  # for marginal_gain
```

---

## Step 2: Clean up `axolotl/collision.py`

### 2a. Remove all commented-out code blocks
Specific locations (line numbers approximate):
- **Lines 96-100**: commented-out weight/rms code inside `best_shift_ei`
- **Lines 145-178**: large commented-out block inside `quick_unit_filter` (alternative alignment approaches, debug prints for unit_134)
- **Lines 190-191**: commented-out `'rms_pre'`, `'rms_post'` keys in dict literal
- **Lines 232-239**: commented-out lag calculation inside `lag_delta_rms`
- **Lines 267**: commented-out `s = 0.0`
- **Lines 280-288**: commented-out per-channel loop inside `lag_delta_rms`
- **Lines 334-342**: commented-out `tmpl_sum` loop inside `score_active_set`
- **Lines 476-497**: commented-out combo/results block inside `evaluate_local_group`
- **Lines 527-533**: commented-out print statements inside `resolve_snippet`
- **Lines 563-567**: commented-out print statements inside `resolve_snippet`
- **Lines 603-607**: commented-out stuck-channel debug inside `resolve_snippet`
- **Lines 653-657**: commented-out mask/sigma block inside `resolve_snippet`
- **Lines 734-791**: large commented-out `marginal_prune` alternative inside `resolve_snippet`
- **Lines 813-889**: another commented-out `marginal_prune` version
- **Lines 907-917**: commented-out per-unit scoring block
- **Lines 1030-1044**: commented-out debug prints inside `marginal_gain`
- **Lines 1217-1221**: commented-out similarity veto inside `per_channel_hist_otsu_candidates`
- **Lines 1277-1327**: entire commented-out `per_channel_gmm_bimodality` (old version)
- **Lines 1880-2050**: entire commented-out block (`_build_data_mask`, `_peak_channel`, `_best_lag`, `_score_delta`, `find_first_missing_unit`)
- **Lines 1936-1945**: commented-out import + plot inside `_score_delta`

### 2b. Remove unused imports
- `from itertools import product` — never used
- `import itertools` — never used
- `from scipy.signal import correlate` (line 40) — only used by moved functions
- `from .comparison import compare_eis` (line 43) — only used in commented-out code
- `from .plotting import plot_ei_waveforms` (line 41) — only used by `plot_ei_quick` (moved) and one commented-out block

### 2c. Remove duplicate imports scattered mid-file
- Line 1088: `from scipy.signal import correlate` (duplicate)
- Line 1089: `import numpy as np` (duplicate)
- Line 1116: `import numpy as np` (duplicate)
- Line 1117: `from sklearn.decomposition import PCA` (duplicate)
- Line 1118: `from sklearn.cluster import KMeans` (duplicate)
- Line 1120-1124: `from sklearn.mixture import GaussianMixture` ×3 + `import numpy as np` ×2
- Line 1361: `import numpy as np` (duplicate)
- Line 1362: `from .comparison import compare_eis` (duplicate)
- Line 1522-1523: `import numpy as np` + `import matplotlib.pyplot as plt` (duplicate)
- Line 1630: `import numpy as np` (duplicate)
- Line 1830: `from scipy.signal import correlate, correlation_lags` (duplicate)
- Line 1878: `import numpy as np` (duplicate)
- Line 2030: `import numpy as np` (duplicate)
- Line 2240: `import matplotlib.pyplot as plt` (duplicate)
- Line 2310: `import matplotlib.pyplot as plt` + `from matplotlib import cm, colors as mcolors` (duplicate)
- Line 2395: `import matplotlib.pyplot as plt` + `from matplotlib import cm, colors as mcolors` (duplicate)
- Line 2458: `import numpy as np` (duplicate)

### 2d. Remove duplicate function definitions
- `tempered_weights` defined at line ~70 AND line ~340 — keep the first (with docstring), delete the second

### 2e. Remove unused functions (moved to diagnostics)
- `delta_rms` (line 80)
- `best_shift_ei` (line 87)
- `_peak_channel` (line 1092)
- `_best_lag` (line 1096)
- `marginal_gain` (line 1115)
- `per_channel_gmm_bimodality_simple` (line 1317)
- `per_channel_hist_otsu_candidates` (line 1364)
- `compute_global_baseline_mean` (line 1526)
- `ei_similarity` (line 1625)
- `median_ei_adaptive` (line 1635)
- `verify_clusters_recursive` (line 1667)
- `xcorr_spike_times` (line 1832)
- `select_template_channels` (line 2010)
- `main_channel_and_neg_peak` (line 2068)
- `roll_zero_2d` (line 2075)
- `compute_harm_map_noamp` (line 2092)
- `plot_harm_heatmap` (line 2180)
- `plot_spike_delta_summary` (line 2228)
- `_help_harm_by_spike` (line 2243)
- `plot_help_harm_lines` (line 2280)
- `compute_spike_gate` (line 2318)
- `plot_help_harm_scatter_swapped` (line 2398)
- `plot_ei_quick` (line 2459)
- `bimodality_probe` (line 2470)
- `build_delta_prototype` (line 2509)
- `reassign_leftovers_to_existing_templates` (line 2582)
- `score_against_prototype` (line 2638)
- `_sanitize_order` (line 2670)
- `build_ideal_delta` (line 2685)
- `compute_profile_deviation_metrics` (line 2728)
- `plot_deviation_lines` (line 2775)
- `compute_deviation_signals` (line 2823)
- `recursive_bimodal_split` (line 2855)
- `_wilson_lower_bound` (line 2956)
- `lag_metrics_from_res` (line 2965)
- `main_channel_traces` (line 3000)
- `format_lag_metrics` (line 3006)

### 2f. Keep in `collision.py` (actively used in core pipeline)
- `roll_zero`
- `roll_zero_all`
- `tempered_weights` (the one at ~line 70)
- `quick_unit_filter`
- `build_channel_index`
- `lag_delta_rms`
- `scan_unit_lags`
- `score_active_set`
- `beam_combo_search`
- `prune_combo`
- `evaluate_local_group`
- `is_certain`
- `resolve_snippet`
- `accumulate_unit_stats`
- `accept_units`
- `micro_align_units`
- `subtract_overlap_tail`
- `per_channel_gmm_bimodality` (the main one, NOT `_simple`)
- `MAX_W_UNITS` constant

### 2g. Clean imports in `collision.py` (final set)
```python
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .plotting import plot_ei_waveforms
```

---

## Step 3: Update `run_axolotl.py` import

`run_axolotl.py` imports `median_ei_adaptive` from collision.py. After the move, update:
```python
# Before:
from axolotl.collision import median_ei_adaptive

# After:
from axolotl.collision_diagnostics import median_ei_adaptive
```

---

## Step 4: Verify

Run `python -c "from axolotl import collision; from axolotl import collision_diagnostics"` to confirm no import errors.

---

## Expected outcome
- `collision.py`: ~600 lines (down from ~3017), clean core pipeline only
- `collision_diagnostics.py`: ~1800 lines, all diagnostic/analysis/plotting helpers
- Zero commented-out code blocks remaining
- Zero duplicate imports
- Zero unused functions in either file
- All cross-references via proper imports
