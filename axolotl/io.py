# axolotl/io.py
"""
Functions for handling file input and output, including loading raw data,
channel maps, and saving spike sorting results.
"""
import numpy as np
import h5py
import os
import yaml
import tempfile
import glob


class VirtualLitkeArray:
    """
    A Virtual Array that mimics a NumPy memmap for Litke datasets.
    Dynamically loads, slices, and concatenates data from multiple 2-minute
    chunk files on the fly, dropping TTL channels and disconnected electrodes.
    """
    def __init__(self, folder_path: str, n_channels_raw: int, dtype: str = 'int16', 
                 max_samples: int = None, connected_electrodes: list = None):
        self.folder_path = folder_path
        self.n_channels_raw = n_channels_raw
        self.dtype = np.dtype(dtype)
        
        # If no specific connected electrodes provided, assume all except TTL
        if connected_electrodes is not None:
            self.connected_electrodes = connected_electrodes
        else:
            self.connected_electrodes = list(range(n_channels_raw - 1))
            
        # TTL is at index 0, so valid data channels are offset by +1 in the raw array
        self.channel_indices = [i + 1 for i in self.connected_electrodes]
        self.n_channels_out = len(self.channel_indices)
        
        # Scan for all binary chunks and sort them chronologically
        self.bin_files = sorted(glob.glob(os.path.join(folder_path, '*.bin')))
        if not self.bin_files:
            self.bin_files = sorted(glob.glob(os.path.join(folder_path, '*.dat')))
            
        if not self.bin_files:
            raise FileNotFoundError(f"No binary data files found in folder: {folder_path}")

        self.file_lengths = []
        total_samples = 0
        item_size = self.dtype.itemsize
        
        # Calculate exactly how many samples exist across all chunked files
        for f in self.bin_files:
            size_bytes = os.path.getsize(f)
            length = size_bytes // (item_size * n_channels_raw)
            self.file_lengths.append((f, length))
            total_samples += length
            
        self.total_samples = total_samples if max_samples is None else min(total_samples, max_samples)
        self.shape = (self.total_samples, self.n_channels_out)
        self.ndim = 2
        print(f"VirtualLitkeArray initialized: {len(self.bin_files)} files mapped, {self.total_samples:,} valid samples across {self.n_channels_out} connected channels.")

    def __getitem__(self, key):
        # Parse the slicing key to separate time slicing from channel slicing
        if isinstance(key, tuple):
            time_key = key[0]
            chan_key = key[1] if len(key) > 1 else slice(None)
        else:
            time_key = key
            chan_key = slice(None)
            
        is_single_time = isinstance(time_key, int)
        if is_single_time:
            time_key = slice(time_key, time_key + 1)
            
        start, stop, step = time_key.indices(self.total_samples)
        if step is not None and step != 1:
            raise NotImplementedError("VirtualLitkeArray only supports a step size of 1.")
            
        if start >= stop:
            return np.empty((0, self.n_channels_out), dtype=self.dtype)[(slice(None), chan_key)]
            
        data_chunks = []
        current_time = 0
        
        # Identify which files contain the requested time block and read only those
        for fname, length in self.file_lengths:
            file_end = current_time + length
            
            if file_end > start and current_time < stop:
                # Calculate local indices within this specific file
                read_start = max(0, start - current_time)
                read_stop = min(length, stop - current_time)
                
                # Memory map the current chunk file
                # Assumes standard (Time, Channels) multiplexed binary format
                mmap_data = np.memmap(fname, dtype=self.dtype, mode='r').reshape(-1, self.n_channels_raw)
                
                # Extract the valid time window and simultaneously drop the TTL and dead channels
                chunk = mmap_data[read_start:read_stop, self.channel_indices]
                data_chunks.append(chunk)
                
            current_time += length
            if current_time >= stop:
                break
                
        # Stitch across file boundaries if the snippet overlaps multiple chunks
        out_data = np.concatenate(data_chunks, axis=0) if len(data_chunks) > 1 else data_chunks[0]
        
        # Apply the requested channel slicing
        out_data = out_data[:, chan_key]
        
        if is_single_time:
            return out_data[0]
        return out_data
        
    def __len__(self):
        return self.shape[0]


def load_litke_folder(folder_path: str, n_channels_raw: int, dtype: str = 'int16', 
                      max_samples: int = None, connected_electrodes: list = None) -> VirtualLitkeArray:
    """
    Initializes a Virtual Array across a folder of chunked Litke binary files.
    """
    return VirtualLitkeArray(folder_path, n_channels_raw, dtype, max_samples, connected_electrodes)


def load_raw_binary(data_path: str, n_channels: int, dtype: str = 'int16', max_samples: int = None) -> np.ndarray:
    """
    Loads raw binary ephys data from a file into a memory-mapped array.
    This function creates a temporary, writable copy of the data to ensure
    the original raw data file is not modified.

    Parameters
    ----------
    data_path : str
        Path to the raw binary data file (.bin or .dat).
    n_channels : int
        The number of channels in the recording.
    dtype : str
        The data type of the raw file (e.g., 'int16').
    max_samples : int, optional
        Maximum number of samples (time points) to load from the start of the file.
        If None, the entire file is loaded. Defaults to None.

    Returns
    -------
    np.ndarray, shape (T, C)
        The raw data as a writable, memory-mapped NumPy array.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found at: {data_path}")
        
    file_size_bytes = os.path.getsize(data_path)
    item_size = np.dtype(dtype).itemsize
    total_possible_samples = file_size_bytes // (item_size * n_channels)
    
    # Determine the number of samples to actually load
    if max_samples is not None and max_samples > 0:
        total_samples = min(total_possible_samples, max_samples)
        print(f"Loading a subset of {total_samples:,} samples for testing.")
    else:
        total_samples = total_possible_samples
    
    print(f"Memory-mapping {total_samples:,} samples from {data_path}...")
    # Open the original data file as read-only, respecting the sample limit
    original_data = np.memmap(data_path, dtype=dtype, mode='r', shape=(n_channels, total_samples), order='F').T
    
    # Create a temporary file to hold the writable memory-mapped copy
    temp_dir = os.path.dirname(data_path) # Store temp file alongside data
    temp_fp = tempfile.NamedTemporaryFile(suffix=".mmap", dir=temp_dir, delete=False)
    temp_path = temp_fp.name
    temp_fp.close() # Close the file handle so memmap can take over

    print(f"Creating a writable temporary copy at: {temp_path}")
    # Create a new, writable memory-mapped file with the correct (potentially smaller) shape
    writable_data = np.memmap(temp_path, dtype=dtype, mode='w+', shape=original_data.shape, order='C')
    
    # Copy the data from the read-only file to the writable one
    writable_data[:] = original_data[:]
    writable_data.flush() # Ensure data is written to the temporary file
    
    print("Data mapped successfully into a writable copy.")
    return writable_data

def load_channel_map(map_path: str) -> np.ndarray:
    """
    Loads a channel map from a .npy file.

    Parameters
    ----------
    map_path : str
        Path to the .npy channel map file.

    Returns
    -------
    np.ndarray, shape (n_channels, 2)
        The electrode coordinates.
    """
    if not map_path or not os.path.exists(map_path):
        raise FileNotFoundError(f"Channel map file not found at: {map_path}")
    
    print(f"Loading channel map from {map_path}")
    return np.load(map_path)

def load_h5_results(h5_path: str) -> dict:
    """
    Loads previously sorted units from the pipeline's HDF5 output file.

    Parameters
    ----------
    h5_path : str
        Path to the results HDF5 file.

    Returns
    -------
    dict
        A dictionary where keys are unit IDs and values are dicts containing
        'spike_times', 'ei', 'selected_channels', and 'peak_channel'.
    """
    units = {}
    if not os.path.exists(h5_path):
        print(f"Warning: HDF5 results file not found at {h5_path}. Returning empty dictionary.")
        return units

    with h5py.File(h5_path, 'r') as h5:
        for unit_name in h5.keys():
            try:
                group = h5[unit_name]
                unit_id = int(unit_name.split('_')[-1])
                units[unit_id] = {
                    'spike_times': group['spike_times'][()],
                    'ei': group['ei'][()],
                    'selected_channels': group['selected_channels'][()],
                    'peak_channel': group.attrs['peak_channel']
                }
            except Exception as e:
                print(f"Could not load unit {unit_name} from HDF5 file: {e}")
    return units


def save_phy_results(
    output_dir: str,
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    templates: np.ndarray,
    channel_map: np.ndarray,
    config: dict
):
    """
    Saves spike sorting results in a Phy-compatible format.

    Parameters
    ----------
    output_dir : str
        The directory where results will be saved.
    spike_times : np.ndarray, shape (n_spikes,)
        The sample index of each detected spike.
    spike_clusters : np.ndarray, shape (n_spikes,)
        The cluster ID assigned to each spike.
    templates : np.ndarray, shape (n_units, n_samples, n_channels)
        The mean waveform (EI) for each unit.
    channel_map : np.ndarray
        The channel map array.
    config : dict
        The configuration dictionary used for the run.
    """
    print(f"Saving Phy-compatible results to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'spike_times.npy'), spike_times.astype(np.int64))
    np.save(os.path.join(output_dir, 'spike_clusters.npy'), spike_clusters.astype(np.int32))
    np.save(os.path.join(output_dir, 'templates.npy'), templates.astype(np.float32))
    np.save(os.path.join(output_dir, 'channel_map.npy'), channel_map)

    # Save the config file for reproducibility
    with open(os.path.join(output_dir, 'params.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    # --- Create and save cluster_group.tsv ---
    print("Creating cluster_group.tsv...")
    unique_clusters = np.unique(spike_clusters)

    with open(os.path.join(output_dir, 'cluster_group.tsv'), 'w') as f:
        f.write("cluster_id\tgroup\n")  # Write the header
        for cluster_id in unique_clusters:
            f.write(f"{cluster_id}\tgood\n")

    print("Results saved successfully.")