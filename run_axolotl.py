#!/usr/bin/env python3
"""
Axolotl Spike Sorter - Main Pipeline

This script runs the iterative spike sorting and peeling pipeline.
It discovers neuron templates (Electrical Images) from raw data,
removes duplicate units, and saves results in Phy format.

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
from typing import List, Dict, Optional, Tuple
from sklearn.decomposition import PCA

# Internal imports
from axolotl.io import load_raw_binary, load_channel_map, save_phy_results
from axolotl.preprocessing import compute_baselines_int16_deriv_robust, subtract_segment_baselines_int16
from axolotl.detection import estimate_spike_threshold_ram, find_dominant_channel_ram
from axolotl.waveform_utils import extract_snippets_fast_ram, estimate_lags_by_xcorr_ram, check_2d_gap_peaks_valley, median_ei_adaptive
from axolotl.clustering import cluster_spike_waveforms, select_cluster_with_largest_waveform
from axolotl.subtraction import apply_residuals, subtract_pca_cluster_means_ram, subtract_scaled_template_ram
from axolotl.comparison import compare_eis
from axolotl.collision_diagnostics import median_ei_adaptive

# ------------------------------------------------------------------------------
# Helper functions for duplicate detection and merging
# ------------------------------------------------------------------------------

def compute_ei_similarity(ei_a: np.ndarray, ei_b: np.ndarray, p2p_thresh: float = 30.0) -> float:
    """
    Compute cosine similarity between two EIs, considering only channels with P2P above threshold.
    """
    p2p_a = np.ptp(ei_a, axis=1)
    p2p_b = np.ptp(ei_b, axis=1)
    mask = (p2p_a > p2p_thresh) | (p2p_b > p2p_thresh)
    if not mask.any():
        return 0.0
    a = ei_a[mask].ravel()
    b = ei_b[mask].ravel()
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(sim)

def reject_duplicate(new_ei: np.ndarray, existing_eis: List[np.ndarray], threshold: float = 0.85) -> bool:
    """
    Returns True if new_ei is too similar to any existing EI (duplicate).
    """
    for ex_ei in existing_eis:
        sim = compute_ei_similarity(new_ei, ex_ei)
        if sim > threshold:
            return True
    return False

def merge_duplicate_units(
    unit_list: List[Dict],
    similarity_threshold: float = 0.85,
    max_lag: int = 3
) -> List[Dict]:
    """
    Merge units whose EIs are highly similar (post-hoc merging).
    Each dict should contain 'ei', 'spike_times', 'unit_id'.
    Returns a new list of merged units.
    """
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix

    n = len(unit_list)
    if n <= 1:
        return unit_list

    # Build similarity matrix
    sim_mat = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            sim = compute_ei_similarity(unit_list[i]['ei'], unit_list[j]['ei'])
            sim_mat[i, j] = sim_mat[j, i] = sim

    # Create graph adjacency (similarity above threshold)
    adj = sim_mat >= similarity_threshold
    graph = csr_matrix(adj)
    n_components, labels = connected_components(csgraph=graph, directed=False)

    merged_units = []
    for comp in range(n_components):
        indices = np.where(labels == comp)[0]
        if len(indices) == 1:
            merged_units.append(unit_list[indices[0]])
        else:
            # Merge: average EIs (weighted by number of spikes)
            total_spikes = sum(len(unit_list[i]['spike_times']) for i in indices)
            merged_ei = np.zeros_like(unit_list[indices[0]]['ei'], dtype=np.float64)
            merged_spikes = []
            for i in indices:
                weight = len(unit_list[i]['spike_times']) / total_spikes
                merged_ei += weight * unit_list[i]['ei'].astype(np.float64)
                merged_spikes.extend(unit_list[i]['spike_times'])
            merged_spikes = np.sort(np.array(merged_spikes))
            merged_units.append({
                'ei': merged_ei.astype(np.float32),
                'spike_times': merged_spikes,
                'unit_id': indices[0]  # keep first id
            })
    return merged_units

# ------------------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------------------

def main(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # ----- Imports (already done above, but keep for clarity) -----
    from axolotl.io import load_raw_binary, load_channel_map, save_phy_results
    from axolotl.preprocessing import compute_baselines_int16_deriv_robust, subtract_segment_baselines_int16
    from axolotl.detection import estimate_spike_threshold_ram, find_dominant_channel_ram
    from axolotl.waveform_utils import extract_snippets_fast_ram, estimate_lags_by_xcorr_ram, check_2d_gap_peaks_valley, median_ei_adaptive
    from axolotl.clustering import cluster_spike_waveforms, select_cluster_with_largest_waveform
    from axolotl.subtraction import apply_residuals, subtract_pca_cluster_means_ram, subtract_scaled_template_ram

    # ----- Parameters -----
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

    # Duplicate detection parameters (add to config or set defaults)
    duplicate_threshold = config.get('duplicate_threshold', 0.85)
    p2p_thresh_for_similarity = config.get('p2p_thresh_similarity', 30.0)
    do_post_merge = config.get('post_merge_duplicates', True)

    # ----- Derived paths -----
    os.makedirs(output_dir, exist_ok=True)
    h5_out_path = os.path.join(output_dir, "axolotl_results.h5")
    baseline_path = os.path.join(output_dir, "baselines.json")

    print(f"Output directory '{output_dir}' is ready.")

    # ----- Test mode handling -----
    max_samples_to_load = None
    if config['testing'].get('enabled', False):
        duration_sec = config['testing']['duration_sec']
        print(f"--- RUNNING IN TEST MODE on first {duration_sec}s of data ---")
        max_samples_to_load = int(duration_sec * sampling_rate)
        max_units_to_find = config['testing'].get('max_units', 15)

    # ----- Load data and channel map -----
    raw_data = load_raw_binary(raw_data_path, n_channels, dtype, max_samples=max_samples_to_load)
    total_samples = raw_data.shape[0]
    ei_positions = load_channel_map(channel_map_path)

    # ----- Baseline computation or loading -----
    if os.path.exists(baseline_path):
        print(f"Loading pre-computed baselines from {baseline_path}")
        with open(baseline_path, 'r') as f:
            data = json.load(f)
        baselines = np.array(data['baselines'], dtype=np.float32)
    else:
        print(f"Computing baselines...")
        baselines = compute_baselines_int16_deriv_robust(raw_data, segment_len=segment_len, diff_thresh=10, trim_fraction=0.15)
        with open(baseline_path, 'w') as f:
            json.dump({'baselines': baselines.tolist()}, f)

    print("Subtracting baselines from raw data...")
    subtract_segment_baselines_int16(raw_data=raw_data, baselines_f32=baselines, segment_len=segment_len)
    print("Preprocessing complete.")

    # ----- Initial threshold estimation -----
    print("Estimating initial spike detection thresholds for all channels...")
    thresholds = np.zeros(n_channels, dtype=float)
    samples_for_thresh = min(total_samples, 5_000_000)

    for ch in range(n_channels):
        threshold, _, _ = estimate_spike_threshold_ram(
            raw_data=raw_data,
            ref_channel=ch,
            total_samples_to_read=samples_for_thresh,
            refractory=refractory_period,
        )
        thresholds[ch] = -np.abs(threshold)

    print("Initial thresholds estimated.")

    # ----- Prepare results storage -----
    if os.path.exists(h5_out_path):
        os.remove(h5_out_path)
        print(f"Removed existing results file: {h5_out_path}")

    unit_id = 0
    ax_ei_list: List[np.ndarray] = []          # EIs for each unit
    all_spike_times: List[np.ndarray] = []     # spike times per unit
    all_spike_clusters: List[np.ndarray] = []  # cluster ids per unit
    # For post-merge we also need to keep raw data per unit
    unit_records: List[Dict] = []              # each dict: ei, spike_times, unit_id

    # ----- Iterative peeling loop -----
    while True:
        if unit_id >= max_units_to_find:
            print(f"Reached unit limit of {max_units_to_find}. Stopping.")
            break

        print(f"\n=== Starting search for unit {unit_id} ===")
        start_time = time.time()

        # 1. Find a promising dominant channel that is still active (threshold not zeroed)
        dominant_channels, _ = find_dominant_channel_ram(
            raw_data=raw_data,
            positions=ei_positions,
        )
        ref_channel = -1
        for ch in dominant_channels:
            if thresholds[ch] < 0:   # active channel
                ref_channel = ch
                break

        if ref_channel == -1:
            print("No more dominant, unprocessed channels found. Stopping.")
            break

        # 2. Estimate spikes on this channel
        threshold, initial_spike_times, _ = estimate_spike_threshold_ram(
            raw_data=raw_data,
            ref_channel=ref_channel,
            total_samples_to_read=total_samples,
            refractory=refractory_period,
        )
        print(f"Selected channel: {ref_channel}, Threshold: {threshold:.1f}, Initial spikes: {len(initial_spike_times)}")
        if len(initial_spike_times) < 20:
            thresholds[ref_channel] = 0
            print("Not enough spikes to process. Skipping to next channel.")
            continue

        # 3. Extract snippets and perform initial clustering
        snips_unaligned, valid_unaligned_times = extract_snippets_fast_ram(
            raw_data=raw_data,
            spike_times=initial_spike_times,
            window=window,
            selected_channels=np.arange(n_channels)
        )
        if snips_unaligned.shape[2] < 20:
            thresholds[ref_channel] = 0
            print("Not enough valid snippets after edge removal. Skipping.")
            continue

        ei_initial = median_ei_adaptive(snips_unaligned)
        k_start = min(5, 3 + (len(valid_unaligned_times) - 1) // 3000)
        clusters_pre = cluster_spike_waveforms(snips_unaligned, ei_initial, k_start=k_start)

        # 4. Select dominant cluster
        try:
            _, cluster_indices, _, _ = select_cluster_with_largest_waveform(clusters_pre, ref_channel)
        except ValueError:
            print("Could not select a dominant cluster. Skipping.")
            thresholds[ref_channel] = 0
            continue

        # 5. Purify selected cluster by removing outliers (2D gap check)
        print(f"Purifying dominant cluster of {len(cluster_indices)} spikes...")
        snips_dominant = snips_unaligned[:, :, cluster_indices]
        n_spikes, n_chans, n_samps = snips_dominant.shape[2], snips_dominant.shape[0], snips_dominant.shape[1]
        if n_spikes > 10:
            snips_flat = snips_dominant.transpose(2, 0, 1).reshape(n_spikes, -1)
            pca = PCA(n_components=2)
            pcs_dominant = pca.fit_transform(snips_flat)
            split_result = check_2d_gap_peaks_valley(pcs_dominant)
            if split_result:
                g1_mask, g2_mask = split_result
                main_mask = g1_mask if g1_mask.sum() > g2_mask.sum() else g2_mask
                original_count = len(cluster_indices)
                cluster_indices = cluster_indices[main_mask]
                print(f"Removed outlier sub-cluster. Kept {len(cluster_indices)}/{original_count} spikes.")

        # 6. Align spikes using cross-correlation on reference channel
        cluster_spike_times = valid_unaligned_times[cluster_indices]
        ref_channel_snips = snips_unaligned[ref_channel, :, cluster_indices]
        snips_for_alignment = ref_channel_snips[:, np.newaxis, :]
        lags = estimate_lags_by_xcorr_ram(snippets=snips_for_alignment, peak_channel_idx=0, window=(-5, 10), max_lag=6)
        aligned_spike_times = cluster_spike_times + lags

        # 7. Re-extract aligned snippets
        snips_final, final_valid_times = extract_snippets_fast_ram(
            raw_data=raw_data,
            spike_times=aligned_spike_times,
            selected_channels=np.arange(n_channels),
            window=window,
        )

        # 8. Compute final EI
        final_ei = median_ei_adaptive(snips_final)

        # 9. DUPLICATE REJECTION: compare with existing units
        if len(ax_ei_list) > 0 and reject_duplicate(final_ei, ax_ei_list, threshold=duplicate_threshold):
            print(f"Unit {unit_id} rejected as duplicate (similarity > {duplicate_threshold}).")
            # Mark the reference channel as used to avoid infinite loop
            thresholds[ref_channel] = 0
            continue

        # 10. Accept unit: store results
        ax_ei_list.append(final_ei)
        all_spike_times.append(final_valid_times)
        all_spike_clusters.append(np.full(final_valid_times.shape, unit_id, dtype=np.int32))
        unit_records.append({'ei': final_ei, 'spike_times': final_valid_times, 'unit_id': unit_id})

        # Optional: Save to HDF5 (diagnostic)
        with h5py.File(h5_out_path, 'a') as h5:
            grp = h5.create_group(f'unit_{unit_id}')
            grp.create_dataset('spike_times', data=final_valid_times, compression='gzip')
            grp.create_dataset('ei', data=final_ei, compression='gzip')
            grp.attrs['peak_channel'] = ref_channel

        # 11. Subtract unit from raw data (peeling)
        p2p_threshold = 30
        ei_p2p = np.ptp(final_ei, axis=1)
        selected_channels_final = np.where(ei_p2p > p2p_threshold)[0]
        if len(final_valid_times) >= 100 and len(selected_channels_final) > 0:
            snips_for_subtraction = snips_final[selected_channels_final, :, :].transpose(2, 0, 1)
            residuals_per_channel = {}
            for ch_idx, ch in enumerate(selected_channels_final):
                ch_snips = snips_for_subtraction[:, ch_idx, :]
                # Use scaled template subtraction for better removal
                template = final_ei[ch, :]
                residuals = subtract_scaled_template_ram(ch_snips, template)
                residuals_per_channel[ch] = residuals
            subtraction_residuals = residuals_per_channel
            subtraction_channels = selected_channels_final
        else:
            print(f"Unit {unit_id} has <100 spikes or no strong channels. Using simple subtraction.")
            template = np.mean(snips_final[ref_channel, :, :], axis=1)
            residuals_T = snips_final[ref_channel, :, :].T - template
            subtraction_residuals = {ref_channel: residuals_T.astype(np.int16)}
            subtraction_channels = [ref_channel]

        write_locs = final_valid_times + window[0]
        apply_residuals(
            raw_data=raw_data,
            residual_snips_per_channel=subtraction_residuals,
            write_locs=write_locs,
            selected_channels=subtraction_channels,
            total_samples=total_samples,
            is_ram=True
        )

        # 12. Recompute thresholds on affected channels (all channels where we subtracted)
        recomputed_count = 0
        for ch in subtraction_channels:
            if thresholds[ch] != 0:
                new_thresh, _, _ = estimate_spike_threshold_ram(
                    raw_data=raw_data, ref_channel=ch, total_samples_to_read=samples_for_thresh, refractory=refractory_period
                )
                thresholds[ch] = -np.abs(new_thresh)
                recomputed_count += 1
        # Also zero out the reference channel to avoid immediate re-detection
        thresholds[ref_channel] = 0
        print(f"Recomputed thresholds on {recomputed_count} channels. Zeroed ref channel {ref_channel}.")

        end_time = time.time()
        print(f"Processed unit {unit_id} with {len(final_valid_times)} final spikes in {end_time - start_time:.1f} seconds.")
        unit_id += 1

    print("\nSpike sorting pipeline finished.")

    # ----- Post-hoc merging of duplicate units (optional) -----
    if do_post_merge and len(unit_records) > 0:
        print("Performing post-hoc merging of similar units...")
        merged = merge_duplicate_units(unit_records, similarity_threshold=duplicate_threshold)
        print(f"Original units: {len(unit_records)}, Merged units: {len(merged)}")
        # Rebuild output arrays from merged units
        final_spike_times = np.concatenate([m['spike_times'] for m in merged])
        final_spike_clusters = np.concatenate([np.full(len(m['spike_times']), i, dtype=np.int32) for i, m in enumerate(merged)])
        final_templates = np.stack([m['ei'] for m in merged], axis=0)
        final_templates = np.transpose(final_templates, (0, 2, 1))  # (unit, samples, channel)
        # Save Phy results
        save_phy_results(
            output_dir=output_dir,
            spike_times=final_spike_times,
            spike_clusters=final_spike_clusters,
            templates=final_templates,
            channel_map=ei_positions,
            config=config
        )
    else:
        # No merging, just aggregate as before
        if unit_id > 0:
            final_spike_times = np.concatenate(all_spike_times)
            final_spike_clusters = np.concatenate(all_spike_clusters)
            sort_idx = np.argsort(final_spike_times)
            final_spike_times = final_spike_times[sort_idx]
            final_spike_clusters = final_spike_clusters[sort_idx]
            final_templates_untransposed = np.stack(ax_ei_list, axis=0)
            final_templates = np.transpose(final_templates_untransposed, (0, 2, 1))
            save_phy_results(
                output_dir=output_dir,
                spike_times=final_spike_times,
                spike_clusters=final_spike_clusters,
                templates=final_templates,
                channel_map=ei_positions,
                config=config
            )
        else:
            print("No units found, skipping Phy export.")

    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Axolotl spike sorting pipeline.")
    parser.add_argument('--config', required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()
    main(args.config)