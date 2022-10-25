import os
import pandas as pd
import numpy as np
import soundfile as sf

# The noisy mixtures are clipped to 0.9
MAX_AMP = 0.9

def read_sources(row, source_dir, noise_dir, n_src):
    # Get information
    mixture_id = row['mixture_ID']
    sources_path_list = get_list_from_csv(row, 'source_path', n_src)
    gain_list = get_list_from_csv(row, 'source_gain', n_src)
    start_time_list = get_list_from_csv(row, 'source_start', n_src)

    # Read the sources
    sources_list = []
    for source_path in sources_path_list:
        source_path = os.path.join(source_dir, source_path)
        source, _ = sf.read(source_path, dtype='float32')
        sources_list.append(source)

    # Read the noise
    noise_path = os.path.join(noise_dir, row['noise_path'])
    noise, _ = sf.read(noise_path, dtype='float32')
    if len(noise.shape) > 1:
        noise = noise[:, 0]
    sources_list.append(noise)
    gain_list.append(row['noise_gain'])
    start_time_list.append(row['noise_start'])
    mixture_length = int(row['audio_length'])
    return mixture_id, gain_list, sources_list, start_time_list, mixture_length

def get_list_from_csv(row, column, n_src):
    # Get all the sources paths
    lst = []
    for i in range(n_src):
        column_name = column.split('_')
        column_name.insert(1, str(i+1))
        column_name = '_'.join(column_name)
        lst.append(row[column_name])
    return lst

def transform_sources(sources_list, gain_list, padding_list, n_src, mixture_length):
    # Get the loudness that was set in the metadata file
    sources_list_norm = loudness_normalize(sources_list, gain_list)
    # TODO: can do resample according to different frequency if other frequency outputs are needed
    # Get the padding that was set in the metadata file
    for i in range(n_src):
        sources_list_norm[i] = sources_list_norm[i][:mixture_length]
    sources_list_norm[-1] = sources_list[-1][int(padding_list[-1]) : int(padding_list[-1]) + mixture_length]
    return sources_list_norm

def loudness_normalize(sources_list, gain_list):
    normalized_list = []
    for i, source in enumerate(sources_list):
        normalized_list.append(source * gain_list[i])
    return normalized_list

def mix_sources(sources_list, n_src):
    if sources_list[0].ndim == 1:
        mixture_length = len(sources_list[0])
    elif sources_list[0].ndim == 2:
        mixture_length = sources_list[0].shape
    else:
        raise NotImplementedError
    mixture = np.zeros(mixture_length)
    for i in range(n_src):
        mixture += sources_list[i]
    return mixture

def mix_noise(mixture, noise):
    mixture_length = len(mixture)
    noise_length = len(noise)
    # TODO: LibriMix provides a way to extend noise, check the script
    # We want noise to be able to cover the whole mixture, but it's not implemented successfully yet -> for now, do right padding first
    if noise_length < mixture_length:
        mixture += np.pad(noise, (0, mixture_length - noise_length), mode='constant')
    else:
        mixture += noise[:mixture_length]
    return mixture

def check_for_clipping(mixture):
    # Check clipping & renormalize if needec
    # TODO: is this normalization correct for 2 channels?
    if np.max(np.abs(mixture)) > MAX_AMP:
        weight = MAX_AMP / np.max(np.abs(mixture))
        mixture = mixture * weight
    return mixture
