import yaml
import argparse
import numpy as np
import os
import h5py
import time
import json
import importlib
from sklearn.decomposition import PCA


# --- Main execution function to encapsulate the pipeline ---
def main(config_path):
    """
    Loads configuration, data, and runs the main spike sorting pipeline.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- Imports and Setup ---
    # Functions are imported from their specific modules within the axolotl package
    from axolotl.io import load_raw_binary, load_channel_map, save_phy_results, load_h5_results
    from axolotl.preprocessing import compute_baselines_int16_deriv_robust, subtract_segment_baselines_int16
    from axolotl.detection import estimate_spike_threshold_ram, find_dominant_channel_ram
    from axolotl.waveform_utils import extract_snippets_fast_ram, estimate_lags_by_xcorr_ram, check_2d_gap_peaks_valley
    from axolotl.clustering import cluster_spike_waveforms, select_cluster_with_largest_waveform
    from axolotl.subtraction import apply_residuals, subtract_pca_cluster_means_ram
    from axolotl.comparison import compare_eis
    from axolotl.collision import median_ei_adaptive

    # --- Use parameters from the loaded config dictionary ---
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

    # --- Derived Paths ---
    os.makedirs(output_dir, exist_ok=True)
    h5_out_path = os.path.join(output_dir, "axolotl_results.h5")
    baseline_path = os.path.join(output_dir, "baselines.json")

    print(f"Output directory '{output_dir}' is ready.")


    # run_axolotl.py in main()

    # --- Determine max samples for test mode BEFORE loading ---
    max_samples_to_load = None
    if config['testing'].get('enabled', False): #<-- CHECKS BEFORE LOADING
        duration_sec = config['testing']['duration_sec']
        print(f"--- RUNNING IN TEST MODE on first {duration_sec}s of data ---")
        max_samples_to_load = int(duration_sec * sampling_rate) #<-- CALCULATES LIMIT
        max_units_to_find = config['testing'].get('max_units', 15)

    # --- Load Data and Preprocess ---
    raw_data = load_raw_binary( #<-- LOADS ONLY WHAT'S NEEDED
        raw_data_path, 
        n_channels, 
        dtype, 
        max_samples=max_samples_to_load  #<-- PASSES THE LIMIT
    )
    total_samples = raw_data.shape[0]

    # The old slicing block is now gone because it's no longer needed.

    ei_positions = load_channel_map(channel_map_path)

    if os.path.exists(baseline_path):
        print(f"Loading pre-computed baselines from {baseline_path}")
        with open(baseline_path, 'r') as f:
            data = json.load(f)
        baselines = np.array(data['baselines'], dtype=np.float32)
    else:
        print(f"Computing baselines...")
        baselines = compute_baselines_int16_deriv_robust(raw_data, segment_len=segment_len, diff_thresh=10, trim_fraction=0.15)
        print(f"Saving baselines to {baseline_path}")
        with open(baseline_path, 'w') as f:
            json.dump({'baselines': baselines.tolist()}, f)

    print("Subtracting baselines from raw data...")
    subtract_segment_baselines_int16(raw_data=raw_data, baselines_f32=baselines, segment_len=segment_len)
    print("Preprocessing complete.")

    # --- Initial Threshold Estimation ---
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

    # --- Iterative Spike Sorting and Peeling ---
    if os.path.exists(h5_out_path):
        os.remove(h5_out_path)
        print(f"Removed existing results file: {h5_out_path}")

    unit_id = 0
    ax_ei_list = []
    
    # ADDED: Initialize lists to collect data for final Phy export
    all_spike_times = []
    all_spike_clusters = []

    while True:
        if unit_id >= max_units_to_find:
            print(f"Reached unit limit of {max_units_to_find}. Stopping.")
            break

        print(f"\n=== Starting search for unit {unit_id} ===")
        start_time = time.time()

        # 1. Find a promising, unprocessed channel to start with
        dominant_channels, _ = find_dominant_channel_ram(
            raw_data=raw_data,
            positions=ei_positions,
            # You can add other parameters from the function signature if needed
        )

        ref_channel = -1
        for ch in dominant_channels:
            # Check if this channel is still active (threshold has not been zeroed out)
            if thresholds[ch] < 0:
                ref_channel = ch
                break # Found our channel for this iteration

        # If no dominant channels are left to process, stop the loop
        if ref_channel == -1:
            print("No more dominant, unprocessed channels found. Stopping.")
            break

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

        # 2. Extract snippets and perform initial clustering
        snips_unaligned, valid_unaligned_times = extract_snippets_fast_ram(
            raw_data=raw_data,
            spike_times=initial_spike_times,
            window=window,
            selected_channels=np.arange(n_channels)
        )
        if snips_unaligned.shape[2] < 20:
            thresholds[ref_channel] = 0
            print("Not enough valid snippets after edge removal. Skipping to next channel.")
            continue

        ei_initial = median_ei_adaptive(snips_unaligned)
        k_start = min(5, 3 + (len(valid_unaligned_times) - 1) // 3000)
        clusters_pre = cluster_spike_waveforms(snips_unaligned, ei_initial, k_start=k_start)

        # 3. Select dominant cluster and align its spikes
        try:
            _, cluster_indices, _, _ = select_cluster_with_largest_waveform(clusters_pre, ref_channel)
        except ValueError:
            print("Could not select a dominant cluster. Skipping to next channel.")
            thresholds[ref_channel] = 0
            continue

        # --- ADDED: Purify the selected cluster by removing outliers ---
        print(f"Purifying dominant cluster of {len(cluster_indices)} spikes...")

        # Get the snippets and compute PCA scores for just this cluster
        snips_dominant = snips_unaligned[:, :, cluster_indices]
        n_spikes, n_chans, n_samps = snips_dominant.shape[2], snips_dominant.shape[0], snips_dominant.shape[1]

        # Ensure there are enough spikes for PCA
        if n_spikes > 10:
            snips_flat = snips_dominant.transpose(2, 0, 1).reshape(n_spikes, -1)

            # We only need 2 components for the 2D check
            pca = PCA(n_components=2)
            pcs_dominant = pca.fit_transform(snips_flat)

            # Check for a separable sub-cluster
            split_result = check_2d_gap_peaks_valley(pcs_dominant)

            if split_result:
                g1_mask, g2_mask = split_result
                # Keep the larger of the two sub-clusters
                if g1_mask.sum() > g2_mask.sum():
                    main_mask = g1_mask
                else:
                    main_mask = g2_mask

                original_count = len(cluster_indices)
                # Update cluster_indices to keep only spikes from the main group
                cluster_indices = cluster_indices[main_mask]
                print(f"Found and removed outlier sub-cluster. Kept {len(cluster_indices)}/{original_count} spikes.")
        # --- END of added purification block ---

        # The rest of the code proceeds from here with the cleaned cluster_indices
        cluster_spike_times = valid_unaligned_times[cluster_indices]
        
        ref_channel_snips = snips_unaligned[ref_channel, :, cluster_indices]
        snips_for_alignment = ref_channel_snips[:, np.newaxis, :]
        
        lags = estimate_lags_by_xcorr_ram(snippets=snips_for_alignment, peak_channel_idx=0, window=(-5, 10), max_lag=6)
        aligned_spike_times = cluster_spike_times + lags

        # 4. Re-extract aligned snippets
        snips_final, final_valid_times = extract_snippets_fast_ram(
            raw_data=raw_data,
            spike_times=aligned_spike_times,
            selected_channels=np.arange(n_channels),
            window=window,
        )
        
        # 5. Compute final EI and save unit
        final_ei = median_ei_adaptive(snips_final)
        ax_ei_list.append(final_ei)

        # ADDED: Collect spike times and cluster IDs for this unit
        all_spike_times.append(final_valid_times)
        all_spike_clusters.append(np.full(final_valid_times.shape, unit_id, dtype=np.int32))

        # (The HDF5 saving can be considered temporary/diagnostic and could be removed later)
        with h5py.File(h5_out_path, 'a') as h5:
            # ... h5 saving logic ...
            pass
            
        # 6. Subtract unit from raw data (Peeling)
        p2p_threshold = 30
        ei_p2p = np.ptp(final_ei, axis=1)
        selected_channels_final = np.where(ei_p2p > p2p_threshold)[0]
        subtraction_residuals = {}
        subtraction_channels = selected_channels_final
        if len(final_valid_times) >= 100 and len(subtraction_channels) > 0:
            snips_for_subtraction = snips_final[selected_channels_final, :, :].transpose(2, 0, 1)
            residuals_per_channel = {}
            for ch_idx, ch in enumerate(selected_channels_final):
                ch_snips = snips_for_subtraction[:, ch_idx, :]
                residuals, _, _ = subtract_pca_cluster_means_ram(
                    snippets=ch_snips, 
                    baselines=baselines[ch, :]*0, 
                    spike_times=final_valid_times,
                    segment_len=segment_len
                )
                residuals_per_channel[ch] = residuals
            subtraction_residuals = residuals_per_channel
        else: # Fallback for small units
            print(f"Unit {unit_id} has < 100 spikes or no channels above P2P threshold. Using simple template subtraction.")
            template = np.mean(snips_final[ref_channel, :, :], axis=1)
            residuals_T = snips_final[ref_channel, :, :].T - template
            subtraction_residuals = {ref_channel: residuals_T.astype(np.int16)}
            subtraction_channels = [ref_channel]
            
        write_locs = final_valid_times + window[0]
        apply_residuals(
            raw_data=raw_data, residual_snips_per_channel=subtraction_residuals, 
            write_locs=write_locs, selected_channels=subtraction_channels, total_samples=total_samples, is_ram=True
        )
        
        # 7. Re-estimate thresholds on affected channels
        recomputed_count = 0
        for ch in range(n_channels):
            neg_amp = -np.min(final_ei[ch, :])
            if thresholds[ch] != 0 and neg_amp >= 0.9 * abs(thresholds[ch]):
                new_thresh, _, _ = estimate_spike_threshold_ram(
                    raw_data=raw_data, ref_channel=ch, total_samples_to_read=samples_for_thresh, refractory=refractory_period
                )
                thresholds[ch] = -np.abs(new_thresh)
                recomputed_count += 1
        print(f"Recomputed thresholds on {recomputed_count} channels.")
        
        # 8. Increment and repeat
        end_time = time.time()
        print(f"Processed unit {unit_id} with {len(final_valid_times)} final spikes in {end_time - start_time:.1f} seconds.")
        unit_id += 1

    print("\nSpike sorting pipeline finished.")

    # ADDED: Final aggregation and saving to Phy format
    print("\nAggregating results for Phy export...")
    if unit_id > 0:
        # Concatenate all spikes and clusters from the lists
        final_spike_times = np.concatenate(all_spike_times)
        final_spike_clusters = np.concatenate(all_spike_clusters)

        # Sort spikes by time, which is required for many tools
        sort_idx = np.argsort(final_spike_times)
        final_spike_times = final_spike_times[sort_idx]
        final_spike_clusters = final_spike_clusters[sort_idx]

        # Stack all the templates (EIs). The templates need to be in (unit, samples, channel) format for Phy.
        # Our EIs are (channel, samples), so we stack and transpose.
        final_templates_untransposed = np.stack(ax_ei_list, axis=0) # Shape: (unit, channel, samples)
        final_templates = np.transpose(final_templates_untransposed, (0, 2, 1)) # Shape: (unit, samples, channel)

        # Call the saving function from io.py
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Axolotl spike sorting pipeline.")
    parser.add_argument('--config', required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()
    main(args.config)