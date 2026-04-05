# axolotl/io.py
"""
Functions for handling file input and output, including loading raw data,
channel maps, and saving spike sorting results.
"""
import atexit
import glob
import os
import tempfile
import yaml

import h5py
import numpy as np


# Track temp files globally so we can clean them up on exit
_TEMP_FILES_TO_CLEANUP = []

def _cleanup_temp_files():
    for path in _TEMP_FILES_TO_CLEANUP:
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"Cleaned up temp file: {path}")
        except Exception as e:
            print(f"Warning: could not remove temp file {path}: {e}")

atexit.register(_cleanup_temp_files)


class VirtualLitkeArray:
    """
    A Virtual Array that mimics a NumPy memmap for Litke datasets.
    Dynamically loads, slices, and concatenates data from multiple 2-minute
    chunk files on the fly, dropping the TTL channel and disconnected electrodes.

    Litke binary files are stored in (Channels, Time) order — channel-major.
    TTL is row 0 in the raw file. Electrode indices passed via
    `connected_electrodes` are 0-based electrode IDs; internally they are
    offset by +1 to skip the TTL row.

    The array presents data in (Time, Channels) order to match the rest of
    the pipeline.
    """
    def __init__(
        self,
        folder_path: str,
        n_channels_raw: int,
        dtype: str = 'int16',
        max_samples: int = None,
        connected_electrodes: list = None
    ):
        self.folder_path = folder_path
        self.n_channels_raw = n_channels_raw
        self.dtype = np.dtype(dtype)

        # connected_electrodes are 0-based electrode IDs (not raw row indices).
        # If not provided, assume all electrodes except TTL are connected.
        if connected_electrodes is not None:
            self.connected_electrodes = list(connected_electrodes)
        else:
            # n_channels_raw includes TTL, so there are (n_channels_raw - 1) electrodes
            self.connected_electrodes = list(range(n_channels_raw - 1))

        # TTL occupies raw row 0; electrode i occupies raw row i+1
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

        # Calculate exactly how many time samples exist in each chunk file.
        # Raw layout is (n_channels_raw, n_samples), so:
        #   n_samples = file_size_bytes / (itemsize * n_channels_raw)
        for f in self.bin_files:
            size_bytes = os.path.getsize(f)
            length = size_bytes // (item_size * n_channels_raw)
            self.file_lengths.append((f, length))
            total_samples += length

        self.total_samples = total_samples if max_samples is None else min(total_samples, max_samples)
        self.shape = (self.total_samples, self.n_channels_out)
        self.ndim = 2
        print(
            f"VirtualLitkeArray initialized: {len(self.bin_files)} files mapped, "
            f"{self.total_samples:,} valid samples across {self.n_channels_out} connected channels."
        )

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
                # Calculate local time indices within this specific file
                read_start = max(0, start - current_time)
                read_stop = min(length, stop - current_time)

                # Litke native format is (Channels, Time) — channel-major.
                # We memmap with that shape, index channels first, then time,
                # and transpose to produce (Time, Channels) output.
                mmap_data = np.memmap(
                    fname, dtype=self.dtype, mode='r',
                    shape=(self.n_channels_raw, length)
                )

                # Select desired electrode rows and time columns, then transpose to (T, C)
                chunk = mmap_data[self.channel_indices, read_start:read_stop].T
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


def load_litke_folder(
    folder_path: str,
    n_channels_raw: int,
    dtype: str = 'int16',
    max_samples: int = None,
    connected_electrodes: list = None
) -> np.memmap:
    """
    Initializes a VirtualLitkeArray across a folder of chunked Litke binary
    files, then copies it into a writable temporary memmap for safe pipeline
    processing.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing Litke .bin or .dat chunk files.
    n_channels_raw : int
        Total number of raw channels including the TTL channel (e.g. 513 for
        a 512-electrode array).
    dtype : str
        Data type of the raw files (default 'int16').
    max_samples : int, optional
        Maximum number of time samples to load. Loads all if None.
    connected_electrodes : list, optional
        0-based electrode IDs to include. Includes all non-TTL electrodes if
        None.

    Returns
    -------
    np.memmap, shape (T, C)
        Writable memory-mapped array in (Time, Channels) order.
    """
    # 1. Map the read-only chunks
    virt_array = VirtualLitkeArray(folder_path, n_channels_raw, dtype, max_samples, connected_electrodes)

    # 2. Create a temporary writable memmap alongside the data
    print("Creating a writable temporary copy of Litke chunks...")
    temp_fp = tempfile.NamedTemporaryFile(suffix=".mmap", dir=folder_path, delete=False)
    temp_path = temp_fp.name
    temp_fp.close()
    _TEMP_FILES_TO_CLEANUP.append(temp_path)

    # 3. Create the writable memmap with (T, C) layout
    writable_data = np.memmap(
        temp_path, dtype=virt_array.dtype, mode='w+',
        shape=virt_array.shape, order='C'
    )

    # 4. Copy data in chunks to avoid blowing up RAM
    chunk_size = 100_000
    for start in range(0, virt_array.total_samples, chunk_size):
        end = min(start + chunk_size, virt_array.total_samples)
        writable_data[start:end] = virt_array[start:end]

    writable_data.flush()
    print(f"Litke data mapped successfully into a writable copy at: {temp_path}")

    # Attach the output channel count so callers can read it
    writable_data.n_channels_out = virt_array.n_channels_out

    return writable_data


def load_raw_binary(
    data_path: str,
    n_channels: int,
    dtype: str = 'int16',
    max_samples: int = None
) -> np.memmap:
    """
    Loads raw binary ephys data (YASS-converted, time-major layout) from a
    single file into a writable memory-mapped array.

    The file on disk is expected to be (Time, Channels) / C-order, which is
    what convert_join_litke_datasets_for_yass.py produces after transposing
    and writing with .tofile().

    Parameters
    ----------
    data_path : str
        Path to the raw binary data file (.bin or .dat).
    n_channels : int
        The number of channels in the recording (TTL already excluded).
    dtype : str
        The data type of the raw file (e.g., 'int16').
    max_samples : int, optional
        Maximum number of samples (time points) to load from the start of
        the file. If None, the entire file is loaded.

    Returns
    -------
    np.memmap, shape (T, C)
        The raw data as a writable, memory-mapped NumPy array.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found at: {data_path}")

    file_size_bytes = os.path.getsize(data_path)
    item_size = np.dtype(dtype).itemsize
    total_possible_samples = file_size_bytes // (item_size * n_channels)

    if max_samples is not None and max_samples > 0:
        total_samples = min(total_possible_samples, max_samples)
        print(f"Loading a subset of {total_samples:,} samples for testing.")
    else:
        total_samples = total_possible_samples

    print(f"Memory-mapping {total_samples:,} samples from {data_path}...")

    # File on disk is (Time, Channels) C-order — memmap directly in that shape
    original_data = np.memmap(
        data_path, dtype=dtype, mode='r',
        shape=(total_samples, n_channels), order='C'
    )

    # Create a temporary writable copy alongside the data file
    temp_dir = os.path.dirname(data_path)
    temp_fp = tempfile.NamedTemporaryFile(suffix=".mmap", dir=temp_dir, delete=False)
    temp_path = temp_fp.name
    temp_fp.close()
    _TEMP_FILES_TO_CLEANUP.append(temp_path)

    print(f"Creating a writable temporary copy at: {temp_path}")
    writable_data = np.memmap(
        temp_path, dtype=dtype, mode='w+',
        shape=original_data.shape, order='C'
    )

    writable_data[:] = original_data[:]
    writable_data.flush()

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
    amplitudes: np.ndarray,
    channel_positions: np.ndarray,
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
    amplitudes : np.ndarray, shape (n_spikes,)
        The amplitude scaling factor for each spike.
    channel_positions : np.ndarray, shape (n_channels, 2)
        The (x, y) electrode coordinates — used as the Phy channel map.
    config : dict
        The configuration dictionary used for the run.
    """
    print(f"Saving Phy-compatible results to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'spike_times.npy'), spike_times.astype(np.int64))
    np.save(os.path.join(output_dir, 'spike_clusters.npy'), spike_clusters.astype(np.int32))
    np.save(os.path.join(output_dir, 'templates.npy'), templates.astype(np.float32))
    np.save(os.path.join(output_dir, 'amplitudes.npy'), amplitudes.astype(np.float32))
    np.save(os.path.join(output_dir, 'channel_map.npy'), channel_positions)

    # Save params.py for Phy — must be a Python file with specific variables
    dat_path = config.get('data_path', 'data.bin')
    n_channels = config.get('n_channels', channel_positions.shape[0])
    sample_rate = config.get('sample_rate', 20000)
    dtype = config.get('dtype', 'int16')
    with open(os.path.join(output_dir, 'params.py'), 'w') as f:
        f.write(f"dat_path = '{dat_path}'\n")
        f.write(f"n_channels_dat = {n_channels}\n")
        f.write(f"dtype = '{dtype}'\n")
        f.write(f"offset = 0\n")
        f.write(f"sample_rate = {sample_rate}\n")
        f.write(f"hp_filtered = True\n")

    # Also save the full config as a yml for reproducibility
    with open(os.path.join(output_dir, 'axolotl_config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Create and save cluster_group.tsv
    print("Creating cluster_group.tsv...")
    unique_clusters = np.unique(spike_clusters)
    with open(os.path.join(output_dir, 'cluster_group.tsv'), 'w') as f:
        f.write("cluster_id\tgroup\n")
        for cluster_id in unique_clusters:
            f.write(f"{cluster_id}\tgood\n")

    print("Results saved successfully.")