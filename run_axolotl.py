#!/usr/bin/env python3
"""
Axolotl Spike Sorter - Main Pipeline

Usage:
    python run_axolotl.py --config config.yaml
"""

import yaml
import argparse
import numpy as np
import os
import h5py
import time
import json
from typing import List, Dict
from sklearn.decomposition import PCA

from axolotl.io import load_raw_binary, load_channel_map, save_phy_results
from axolotl.preprocessing import compute_baselines_int16_deriv_robust, subtract_segment_baselines_int16
from axolotl.detection import estimate_spike_threshold_ram, find_dominant_channel_ram
from axolotl.waveform_utils import extract_snippets_fast_ram, estimate_lags_by_xcorr_ram, check_2d_gap_peaks_valley
from axolotl.clustering import cluster_spike_waveforms, select_cluster_with_largest_waveform
from axolotl.subtraction import apply_residuals, subtract_scaled_template_ram
from axolotl.comparison import compare_eis
from axolotl.collision_diagnostics import median_ei_adaptive

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def reject_duplicate(new_ei: np.ndarray, existing_eis: List[np.ndarray],
                     threshold: float = 0.85, max_lag: int = 3) -> bool:
    """Returns True if new_ei is too similar to any existing EI (lag-tolerant)."""
    if not existing_eis:
        return False
    sims = compare_eis(existing_eis + [new_ei], max_lag=max_lag)[-1, :-1]
    return bool(np.any(sims > threshold))


def merge_duplicate_units(unit_list: List[Dict], similarity_threshold: float = 0.85,
                          max_lag: int = 3) -> List[Dict]:
    """Merge units whose EIs are highly similar (post-hoc, lag-tolerant)."""
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix

    n = len(unit_list)
    if n <= 1:
        return unit_list

    eis = [u['ei'] for u in unit_list]
    sim_mat = compare_eis(eis, max_lag=max_lag)
    graph = csr_matrix(sim_mat >= similarity_threshold)
    n_components, labels = connected_components(csgraph=graph, directed=False)

    merged_units = []
    for comp in range(n_components):
        indices = np.where(labels == comp)[0]
        if len(indices) == 1:
            merged_units.append(unit_list[indices[0]])
        else:
            total_spikes = sum(len(unit_list[i]['spike_times']) for i in indices)
            merged_ei = np.zeros_like(unit_list[indices[0]]['ei'], dtype=np.float64)
            merged_spikes = []
            for i in indices:
                weight = len(unit_list[i]['spike_times']) / total_spikes
                merged_ei += weight * unit_list[i]['ei'].astype(np.float64)
                merged_spikes.extend(unit_list[i]['spike_times'])
            merged_units.append({
                'ei': merged_ei.astype(np.float32),
                'spike_times': np.sort(np.array(merged_spikes)),
                'unit_id': indices[0],
            })
    return merged_units

# ------------------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------------------

def main(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    raw_data_path = config['paths']['raw_data_path']
    channel_map_path = config['paths']['channel_map_path']
    output_dir = config['paths']['output_dir']

    n_channels = config['recording']['n_channels']
    sampling_rate = config['recording']['sampling_rate']
    dtype = config['recording']['dtype']

    max_units_to_find = config['pipeline']['max_units_to_find']
    window = (config['pipeline']['window_pre_samples'], config['pipeline']['window_post_samples'])
    refractory_period = config['pipeline']['refractory_samples']
    segment_len = config['preprocessing']['segment_len']

    duplicate_threshold = config.get('duplicate_threshold', 0.85)
    do_post_merge = config.get('post_merge_duplicates', True)

    os.makedirs(output_dir, exist_ok=True)
    h5_out_path = os.path.join(output_dir, "axolotl_results.h5")
    baseline_path = os.path.join(output_dir, "baselines.json")

    max_samples_to_load = None
    if config['testing'].get('enabled', False):
        duration_sec = config['testing']['duration_sec']
        print(f"--- TEST MODE: first {duration_sec}s ---")
        max_samples_to_load = int(duration_sec * sampling_rate)
        max_units_to_find = config['testing'].get('max_units', 15)

    raw_data = load_raw_binary(raw_data_path, n_channels, dtype, max_samples=max_samples_to_load)
    total_samples = raw_data.shape[0]
    ei_positions = load_channel_map(channel_map_path)

    if os.path.exists(baseline_path):
        print(f"Loading pre-computed baselines from {baseline_path}")
        with open(baseline_path, 'r') as f:
            baselines = np.array(json.load(f)['baselines'], dtype=np.float32)
    else:
        print("Computing baselines...")
        baselines = compute_baselines_int16_deriv_robust(raw_data, segment_len=segment_len,
                                                         diff_thresh=10, trim_fraction=0.15)
        with open(baseline_path, 'w') as f:
            json.dump({'baselines': baselines.tolist()}, f)

    print("Subtracting baselines...")
    subtract_segment_baselines_int16(raw_data=raw_data, baselines_f32=baselines, segment_len=segment_len)

    # ----- Initial threshold estimation (once, before the loop) -----
    print("Estimating initial thresholds...")
    samples_for_thresh = min(total_samples, 5_000_000)
    thresholds = np.zeros(n_channels, dtype=float)
    for ch in range(n_channels):
        thresh, _, _ = estimate_spike_threshold_ram(raw_data=raw_data, ref_channel=ch,
                                                    total_samples_to_read=samples_for_thresh,
                                                    refractory=refractory_period)
        thresholds[ch] = -np.abs(thresh)

    if os.path.exists(h5_out_path):
        os.remove(h5_out_path)

    unit_id = 0
    ax_ei_list: List[np.ndarray] = []
    all_spike_times: List[np.ndarray] = []
    all_spike_clusters: List[np.ndarray] = []
    unit_records: List[Dict] = []

    # Open HDF5 once for the full run (issue #16)
    with h5py.File(h5_out_path, 'w') as h5:
        while True:
            if unit_id >= max_units_to_find:
                print(f"Reached unit limit of {max_units_to_find}. Stopping.")
                break

            print(f"\n=== Unit {unit_id} ===")
            start_time = time.time()

            # 1. Find promising dominant channel
            dominant_channels, _ = find_dominant_channel_ram(raw_data=raw_data, positions=ei_positions)
            ref_channel = next((ch for ch in dominant_channels if thresholds[ch] < 0), -1)
            if ref_channel == -1:
                print("No more active channels. Stopping.")
                break

            # 2. Use pre-computed threshold to find spikes (issue #9)
            threshold = thresholds[ref_channel]
            _, initial_spike_times, _ = estimate_spike_threshold_ram(
                raw_data=raw_data, ref_channel=ref_channel,
                total_samples_to_read=total_samples, refractory=refractory_period,
            )
            print(f"Channel: {ref_channel}, Threshold: {threshold:.1f}, Spikes: {len(initial_spike_times)}")
            if len(initial_spike_times) < 20:
                thresholds[ref_channel] = 0
                print("Too few spikes. Skipping.")
                continue

            # 3. Extract snippets on all channels (needed for EI / clustering)
            snips_unaligned, valid_unaligned_times = extract_snippets_fast_ram(
                raw_data=raw_data,
                spike_times=initial_spike_times,
                window=window,
                selected_channels=np.arange(n_channels),
            )
            if snips_unaligned.shape[2] < 20:
                thresholds[ref_channel] = 0
                print("Not enough valid snippets. Skipping.")
                continue

            ei_initial = median_ei_adaptive(snips_unaligned)
            k_start = min(5, 3 + (len(valid_unaligned_times) - 1) // 3000)
            clusters_pre = cluster_spike_waveforms(snips_unaligned, ei_initial, k_start=k_start)

            # 4. Select dominant cluster
            try:
                _, cluster_indices, _, _ = select_cluster_with_largest_waveform(clusters_pre, ref_channel)
            except ValueError:
                print("Could not select dominant cluster. Skipping.")
                thresholds[ref_channel] = 0
                continue

            # 5. Purify dominant cluster
            print(f"Purifying dominant cluster of {len(cluster_indices)} spikes...")
            snips_dominant = snips_unaligned[:, :, cluster_indices]
            n_spikes = snips_dominant.shape[2]
            if n_spikes > 10:
                snips_flat = snips_dominant.transpose(2, 0, 1).reshape(n_spikes, -1)
                pcs_dominant = PCA(n_components=2).fit_transform(snips_flat)
                split_result = check_2d_gap_peaks_valley(pcs_dominant)
                if split_result:
                    g1_mask, g2_mask = split_result
                    main_mask = g1_mask if g1_mask.sum() > g2_mask.sum() else g2_mask
                    orig_count = len(cluster_indices)
                    cluster_indices = cluster_indices[main_mask]
                    print(f"Kept {len(cluster_indices)}/{orig_count} spikes after purification.")

            # 6. Align spikes
            cluster_spike_times = valid_unaligned_times[cluster_indices]
            ref_channel_snips = snips_unaligned[ref_channel, :, cluster_indices]
            lags = estimate_lags_by_xcorr_ram(
                snippets=ref_channel_snips[:, np.newaxis, :],
                peak_channel_idx=0, window=(-5, 10), max_lag=6,
            )
            aligned_spike_times = cluster_spike_times + lags

            # 7. Re-extract aligned snippets — only channels with meaningful signal (issue #15)
            ei_p2p_pre = np.ptp(ei_initial, axis=1)
            p2p_threshold = 30
            selected_channels_final = np.where(ei_p2p_pre > p2p_threshold)[0]
            if len(selected_channels_final) == 0:
                selected_channels_final = np.argsort(ei_p2p_pre)[-30:]
            selected_channels_final = np.sort(selected_channels_final)

            snips_final, final_valid_times = extract_snippets_fast_ram(
                raw_data=raw_data,
                spike_times=aligned_spike_times,
                selected_channels=selected_channels_final,
                window=window,
            )

            # 8. Compute final EI on selected channels; build full-array EI for comparison
            ei_selected = median_ei_adaptive(snips_final)
            final_ei = np.zeros((n_channels, ei_selected.shape[1]), dtype=np.float32)
            final_ei[selected_channels_final] = ei_selected

            # 9. Duplicate rejection (lag-tolerant, issue #8)
            if reject_duplicate(final_ei, ax_ei_list, threshold=duplicate_threshold):
                print(f"Unit {unit_id} rejected as duplicate.")
                thresholds[ref_channel] = 0
                continue

            # 10. Accept unit
            ax_ei_list.append(final_ei)
            all_spike_times.append(final_valid_times)
            all_spike_clusters.append(np.full(final_valid_times.shape, unit_id, dtype=np.int32))
            unit_records.append({'ei': final_ei, 'spike_times': final_valid_times, 'unit_id': unit_id})

            grp = h5.create_group(f'unit_{unit_id}')
            grp.create_dataset('spike_times', data=final_valid_times, compression='gzip')
            grp.create_dataset('ei', data=final_ei, compression='gzip')
            grp.attrs['peak_channel'] = ref_channel

            # 11. Subtract unit from raw data
            subtraction_channels = selected_channels_final
            if len(final_valid_times) >= 100 and len(subtraction_channels) > 0:
                snips_for_sub = snips_final.transpose(2, 0, 1)  # (n_spikes, n_sel_ch, T)
                residuals_per_channel = {}
                for ch_idx, ch in enumerate(subtraction_channels):
                    ch_snips = snips_for_sub[:, ch_idx, :]
                    residuals_per_channel[ch] = subtract_scaled_template_ram(ch_snips, ei_selected[ch_idx])
                subtraction_residuals = residuals_per_channel
            else:
                print(f"Unit {unit_id} <100 spikes. Using simple subtraction.")
                template = np.mean(snips_final[0, :, :], axis=1)  # peak channel (index 0 of selected)
                residuals_T = snips_final[0, :, :].T - template
                # Fix: clip before int16 cast (issue #6)
                residuals_T = np.clip(residuals_T, -32768, 32767).astype(np.int16)
                subtraction_residuals = {selected_channels_final[0]: residuals_T}
                subtraction_channels = [selected_channels_final[0]]

            apply_residuals(
                raw_data=raw_data,
                residual_snips_per_channel=subtraction_residuals,
                write_locs=final_valid_times + window[0],
                selected_channels=subtraction_channels,
                total_samples=total_samples,
                is_ram=True,
            )

            # 12. Recompute thresholds on affected channels
            recomputed = 0
            for ch in subtraction_channels:
                if thresholds[ch] != 0:
                    new_thresh, _, _ = estimate_spike_threshold_ram(
                        raw_data=raw_data, ref_channel=ch,
                        total_samples_to_read=samples_for_thresh,
                        refractory=refractory_period,
                    )
                    thresholds[ch] = -np.abs(new_thresh)
                    recomputed += 1
            thresholds[ref_channel] = 0
            print(f"Recomputed {recomputed} thresholds. Zeroed ref={ref_channel}.")

            elapsed = time.time() - start_time
            print(f"Unit {unit_id}: {len(final_valid_times)} spikes in {elapsed:.1f}s.")
            unit_id += 1

    print("\nPipeline finished.")

    # ----- Post-hoc merge -----
    if do_post_merge and len(unit_records) > 0:
        print("Post-hoc merging...")
        merged = merge_duplicate_units(unit_records, similarity_threshold=duplicate_threshold)
        print(f"Units: {len(unit_records)} → {len(merged)} after merge.")
        final_spike_times = np.concatenate([m['spike_times'] for m in merged])
        final_spike_clusters = np.concatenate([
            np.full(len(m['spike_times']), i, dtype=np.int32) for i, m in enumerate(merged)
        ])
        final_templates = np.transpose(np.stack([m['ei'] for m in merged], axis=0), (0, 2, 1))
        save_phy_results(output_dir, final_spike_times, final_spike_clusters,
                         final_templates, ei_positions, config)
    elif unit_id > 0:
        final_spike_times = np.concatenate(all_spike_times)
        final_spike_clusters = np.concatenate(all_spike_clusters)
        sort_idx = np.argsort(final_spike_times)
        final_templates = np.transpose(np.stack(ax_ei_list, axis=0), (0, 2, 1))
        save_phy_results(output_dir, final_spike_times[sort_idx], final_spike_clusters[sort_idx],
                         final_templates, ei_positions, config)
    else:
        print("No units found, skipping Phy export.")

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    main(parser.parse_args().config)