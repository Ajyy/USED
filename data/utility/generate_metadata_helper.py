import os
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import random

random.seed(0)

# Global parameters
# In LibriSpeech all the sources are at 16K Hz
RATE = 16000
# Random loudness is set between -25 to -33
MIN_LOUDNESS = -33
MAX_LOUDNESS = -25
# The noisy mixtures are clipped to 0.9
MAX_AMP = 0.9
# Output mixture has frequency
FREQUENCY = 16000

def remove_duplicates(utt_pairs):
    for i, pair in enumerate(utt_pairs):
        for j, du_pair in enumerate(utt_pairs):
            if sorted(pair) == sorted(du_pair) and i != j:
                utt_pairs.remove(du_pair)
    return utt_pairs

def get_mix_info(sources_list, sources_info, mode, n_src):
    s1_length = int(len(sources_list[0]))
    s2_length = int(len(sources_list[1]))
    mixture_length = int(s1_length + s2_length)

    if mode in ['train']: # we want to generate more SS case so that the training of model is more efficient
        scenarios = [0, 1, 2, 3] # 0: 0-80% overlap; 1: 80-100% overlap; 2: fully overlap; 3: no overlap, adding quiet region
        weights = [0.01, 0.001, 1.2, 0.2] # we manually add in these weights to create more overlap regions for model training

        scenario = random.choices(population=scenarios, weights=weights, k=1)[0]
        if scenario == 0:
            ratio = random.uniform(0, 0.8)
        elif scenario == 1:
            ratio = random.uniform(0.8, 1)
        elif scenario == 2:
            ratio = 1.0
        elif scenario == 3:
            ratio = random.uniform(-0.75, 0)
        else:
            raise NotImplementedError
    elif mode in ['dev', 'test']:
        ratio = random.uniform(-0.3, 1.0)
    else:
        raise NotImplementedError

    if ratio >= 0:
        overlap_length = int(mixture_length * ratio / 2)
        mixture_length = mixture_length - overlap_length
        pad_length1 = s2_length - overlap_length
        pad_length2 = s1_length - overlap_length
    else:
        ratio = -1 * ratio
        overlap_length = int(mixture_length * ratio)
        mixture_length = mixture_length + overlap_length
        pad_length1 = mixture_length - s1_length
        pad_length2 = mixture_length - s2_length

    if pad_length1 < 0:
        pad_length2 = s1_length - s2_length
        pad_length1 = 0
        mixture_length = s1_length

    elif pad_length2 < 0:
        pad_length1 = s2_length - s1_length
        pad_length2 = 0
        mixture_length = s2_length

    padding_list = [[0, pad_length1], [pad_length2, 0]]
    sources_info['padding_list'] = padding_list

    return sources_info, mixture_length

def get_noise(noise_md, noise_dir, sources_list, sources_info, mixture_length):
    possible = noise_md[noise_md['length'] >= mixture_length]
    pad_noise = False
    # If possibile is not empty
    try:
        noise_idx = random.sample(list(possible.index), 1)
    except ValueError:
        # Get the longest in the metadata file, the noise are arranged according to length
        noise_idx = list(noise_md.index)[-1]

    noise = noise_md.loc[noise_idx]
    try:
        noise_path = os.path.join(noise_dir, noise['origin_path'].values[0])
    except:
        noise_path = os.path.join(noise_dir, noise['origin_path'])
    try:
        # TODO: some errors occured in reading of some augmented noise files
        data, _ = sf.read(noise_path, dtype='float32')
    except:
        return None, None, None

    # Keep the first channel
    if len(data.shape) > 1:
        data = data[:, 0]

    # Pad noise if shorter
    if len(data) < mixture_length:
        pad_noise = True
        sources_list.append(np.pad(data, (0, mixture_length - len(data)), mode='constant'))
        sources_info['noise_start'] = 0
    # Cut if longer
    else:
        left_padding = len(data) - mixture_length
        left_padding = random.randint(0, left_padding)
        sources_list.append(data[left_padding: left_padding + mixture_length])
        sources_info['noise_start'] = left_padding
    try:
        sources_info['noise_path'] = noise['origin_path'].values[0]
    except:
        sources_info['noise_path'] = noise['origin_path']
    return sources_info, sources_list, pad_noise

def set_loudness(sources_list):
    # LibriMix randomize the loudness (sampled uniformly between -25 and -33 Loudness units relative to full scall (LUFS)
    # Here we adopt the same strategy
    loudness_list = []
    meter = pyln.Meter(RATE)
    sources_list_norm =[]

    for i in range(len(sources_list)):
        # Compute initial loudness
        loudness_list.append(meter.integrated_loudness(sources_list[i]))
        # Noise loudness is set between -38 and -30
        if i == len(sources_list) - 1:
            target_loudness = random.uniform(MIN_LOUDNESS-5, MAX_LOUDNESS-5)
        # Choose a random loudness
        else:
            target_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)

        # Normalize the audio to target loudness
        src = pyln.normalize.loudness(sources_list[i], loudness_list[i], target_loudness)

        # IF source clips, renomalize
        if np.max(np.abs(src)) >= 1:
            src = sources_list[i] * MAX_AMP / np.max(np.abs(sources_list[i]))
        sources_list_norm.append(src)
    return loudness_list, sources_list_norm

def mix(sources_list, sources_info, n_src, mixture_length):
    mixture = np.zeros(mixture_length)
    for i in range(len(sources_list)-1):
        mixture += np.pad(sources_list[i], (sources_info['padding_list'][i][0], sources_info['padding_list'][i][1]), mode='constant')
    mixture += sources_list[-1]
    return mixture

def check_for_cliping(mixture, sources_list):
    # Renormalized sources and loudness
    renormalize_loudness = []
    clip = False
    meter = pyln.Meter(RATE)
    # Check clipping
    if np.max(np.abs(mixture)) > MAX_AMP:
        clip = True
        weight = MAX_AMP / np.max(np.abs(mixture))
    else:
        weight = 1
    # Renormalize
    for i in range(len(sources_list)):
        new_loudness = meter.integrated_loudness(sources_list[i] * weight)
        renormalize_loudness.append(new_loudness)
    return renormalize_loudness, clip

def compute_gain(loudness, target_loudness):
    gain = []
    for i in range(len(loudness)):
        delta_loudness = target_loudness[i] - loudness[i]
        gain.append(np.power(10.0, delta_loudness / 20.0))
    return gain

def get_row(sources_info, sources_list, gain_list, n_src, mixture_length):
    audio_length = [len(source) for source in sources_list]

    row = [sources_info['mixture_id']]
    for i in range(n_src):
        row.append(sources_info['path_list'][i])
        row.append(sources_info['speaker_id_list'][i])
        row.append(gain_list[i])
        row.append(sources_info['padding_list'][i][0])
        row.append(sources_info['padding_list'][i][0] + audio_length[i])
    row.append(sources_info['noise_path'])
    row.append(gain_list[-1])
    row.append(sources_info['noise_start'])
    row.append(mixture_length)
    return row
