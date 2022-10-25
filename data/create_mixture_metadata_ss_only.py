import argparse
import os
import numpy as np
import pandas as pd
import random
import soundfile as sf
import pyloudnorm as pyln
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from pathlib import Path

from utility.generate_metadata_helper_ss_only import remove_duplicates, get_mix_info, get_noise, set_loudness, mix, check_for_cliping, compute_gain, get_row

# Setting seeds to ensure replicate of the dataset generated
random.seed(0)

# Output mixture has frequency
FREQUENCY=16000

# Pass in arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--data_md_fpath', type=str, required=True)
parser.add_argument('--noise_dir', type=str, required=True)
parser.add_argument('--noise_md_dir', type=str, required=True)
parser.add_argument('--metadata_outdir', type=str, required=True)
parser.add_argument('--n_src', type=int, default=2,
                   help='Number of sources desired to create the mixture. Currently only n_src = 2 is tested')
parser.add_argument('--output_num', type=int, default=None, required=False,
                    help='Specify the amount of data needed to be generated')

def main(args):
    dataset_name = args.dataset_name
    data_dir = args.data_dir # contain all segmented files
    data_md_fpath = args.data_md_fpath # metadata file
    noise_dir = args.noise_dir
    noise_md_dir = args.noise_md_dir
    metadata_outdir = args.metadata_outdir
    n_src = args.n_src
    output_num = args.output_num

    mode = Path(data_md_fpath).stem
    alt_name = mode
    mode = mode.split("-")[0]
    noise_md_file = os.path.join(noise_md_dir, f"all_noises_{mode}.csv")

    mixtures_md = create_mixture_metadata(data_dir, data_md_fpath, noise_dir, noise_md_file, output_num, mode, n_src=n_src)
    mixtures_md = mixtures_md.sort_values(by=['audio_length'])

    # Save metadata
    if alt_name in ['dev', 'test']: # test, dev -> test; train -> train, dev
        alt_name = 'test'
        metadata_save_path = os.path.join(metadata_outdir, f'{dataset_name}_' + alt_name + f'_{n_src}mixture_metadata.csv')
        print(f"Saving metadata to {metadata_save_path}. Be careful we are using appending operation here.")
        if os.path.exists(metadata_save_path):
            df = pd.read_csv(metadata_save_path, engine='python')
            mixtures_md = df.append(mixtures_md)
            mixtures_md = mixtures_md.sort_values(by=['audio_length'])
        mixtures_md.to_csv(metadata_save_path, index=False)
    elif alt_name in ['train', 'train-clean-360']:
        # Random split train to train and dev
        dev_mixtures_md, train_mixtures_md = train_test_split(mixtures_md, test_size=1/1.1, random_state=0)
        assert len(dev_mixtures_md) == 0.1 * len(train_mixtures_md), f"Required split not produced, {len(dev_mixtures_md)} and {len(train_mixtures_md)}"
        train_metadata_save_path = os.path.join(metadata_outdir, f'{dataset_name}_' + alt_name + f'_{n_src}mixture_metadata.csv')
        print(f"Saving metadata to {train_metadata_save_path}")
        train_mixtures_md.to_csv(train_metadata_save_path, index=False)

        # save dev with another name
        alt_name = alt_name.replace("train", "dev")
        dev_metadata_save_path = os.path.join(metadata_outdir, f'{dataset_name}_' + alt_name + f'_{n_src}mixture_metadata.csv')
        print(f"Saviing metadata to {dev_metadata_save_path}. Be careful we are using appending operation here.")
        dev_mixtures_md.to_csv(dev_metadata_save_path, index=False)


def create_mixture_metadata(data_dir, data_md_fpath, noise_dir, noise_md_file, output_num, mode, n_src=2):
    # Read files
    md = pd.read_csv(data_md_fpath, engine='python')
    md = md[md['length'] > 0.4 * 16000]
    md = md.reset_index()
    noise_md = pd.read_csv(noise_md_file, engine='python')

    # Create a dataframe to store metadata
    mixtures_md = pd.DataFrame(columns=['mixture_ID'])
    for i in range(n_src):
        mixtures_md[f'source_{i + 1}_path'] = {}
        mixtures_md[f'source_{i + 1}_speaker_ID'] = {}
        mixtures_md[f'source_{i + 1}_gain'] = {}
        mixtures_md[f'source_{i + 1}_start'] = {}
        mixtures_md[f'source_{i + 1}_end'] = {}
    mixtures_md['noise_path'] = {}
    mixtures_md['noise_gain'] = {}
    mixtures_md['noise_start'] = {}
    mixtures_md['audio_length'] = {}

    utt_pairs = generate_pairs(md, n_src, output_num)
    noise_md_copy = noise_md[noise_md['augmented'] == False].copy()
    if len(utt_pairs) > len(noise_md_copy): #only use augmented mixture if there are more utt_pairs than non-augmented data
        noise_md_copy = noise_md

    clip_counter = 0
    pad_noise_counter = 0

    for utt_pair in tqdm(utt_pairs):
        sources_info, sources_list = read_sources(data_dir, md, utt_pair, n_src) # contains [S1, S2]
        sources_info, mixture_length = get_mix_info(sources_list, sources_info, mode, n_src)
        sources_info, sources_list, pad_noise = get_noise(noise_md_copy, noise_dir, sources_list, sources_info, mixture_length) # contains [S1. S2, noise]
        if pad_noise is None:
            continue
        output_num -= 1
        pad_noise_counter += int(pad_noise)
        loudness_list, sources_list_norm = set_loudness(sources_list) # contains [S1, S2, noise]
        mixture = mix(sources_list_norm, sources_info, n_src, mixture_length)
        renormalize_loudness, did_clip = check_for_cliping(mixture, sources_list_norm)
        clip_counter += int(did_clip)
        gain_list = compute_gain(loudness_list, renormalize_loudness)
        row = get_row(sources_info, sources_list_norm, gain_list, n_src, mixture_length)
        mixtures_md.loc[len(mixtures_md)] = row
        if output_num == 0:
            break
    print(f"Among {output_num}, {clip_counter} clipped, {pad_noise_counter} noises padded.")
    return mixtures_md

def generate_pairs(md, n_src, output_num):
    utt_pairs = []
    counter = 0

    while len(utt_pairs) < output_num:
        counter += 1
        print(f"Did not reach limit. Counter = {counter}. {len(utt_pairs)} out of {output_num} generated")
        utt_pairs = get_utt_pairs(md, utt_pairs, n_src)
        utt_pairs = remove_duplicates(utt_pairs)
    return utt_pairs

def get_utt_pairs(md, utt_pairs, n_src):
    c = 0
    index = list(range(len(md)))

    while check_speaker_remaining(md.loc[index]) and c < 400:
        couple = random.sample(index, n_src)
        speaker_list = set([md.iloc[couple[i]]['speaker'] for i in range(n_src)])
        if len(speaker_list) != n_src:
            c += 1
        else:
            for i in range(n_src):
                index.remove(couple[i])
            utt_pairs.append(couple)
            c = 0
    return utt_pairs

def check_speaker_remaining(md):
    speaker_list = md['speaker'].values
    return len(set(speaker_list)) >= 2

def read_sources(data_dir, md, utt_pair, n_src):
    sources = [md.iloc[utt_pair[i]] for i in range(n_src)]

    speaker_id_list = [source['speaker'] for source in sources]
    path_list = [source['origin_path'] for source in sources]
    id_l = [Path(source['filename']).stem for source in sources]
    mixtures_id = "_".join(id_l)

    sources_list = []
    for i in range(n_src):
        source = md.iloc[utt_pair[i]]
        path = os.path.join(data_dir, source['origin_path'])
        data, freq = sf.read(path, dtype='float32')
        assert len(data) == source['length']
        assert freq == FREQUENCY, f"{freq} != {FREQUENCY}"
        sources_list.append(data)
    sources_info = {'mixture_id': mixtures_id,
                    'speaker_id_list': speaker_id_list,
                    'path_list': path_list}
    print(sources_info)
    return sources_info, sources_list

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


