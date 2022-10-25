"""
This script is created to output mixtures.
You need the following metadata files before running this file:
    - {dataset}_{n_src}mixture_metadata.csv
    - {dataset}_{n_src}_reverb_params.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import soundfile as sf
import pyloudnorm as pyln
import functools
import glob
from tqdm import tqdm
import tqdm.contrib.concurrent as tcc
from pathlib import Path
import time

from utility.create_mixture_helper import read_sources, get_list_from_csv, transform_sources, loudness_normalize, mix_sources, mix_noise, check_for_clipping
from utility.wham_room import WhamRoom

# Output mixture has frequency
FREQUENCY = 16000

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True,
                    help='Root directory of the segmented data')
parser.add_argument('--noise_dir', type=str, required=True,
                    help='Path to all the noise root directory')
parser.add_argument('--dataset', type=str, required=True,
                    help='One of wsj0 or librispeech')
parser.add_argument('--metadata_fpath', type=str, required=True,
                    help='Path to the mixture metadata file')
parser.add_argument('--reverb_fpath', type=str, required=True,
                    help='Path to the reverb metadata file, it shoud corresponds to metadata_fpath')
parser.add_argument('--data_outdir', type=str, default=None,
                    help='Path to the desired dataset root directory')
parser.add_argument('--mixture_type', type=str, default="['clean_sources_anechoic', 'mix_sources', 'reverb_noisy']",
                    help='Type of mixtures generated. It should be a list including any of the '
                         '[clean_sources, mix_sources, anechoic_nosiy, reverb_noisy]')
parser.add_argument('--n_src', type=int, default=2,
                    help='Number of sources in mixtures')

def main(args):
    data_dir = args.data_dir
    noise_dir = args.noise_dir
    dataset = args.dataset
    metadata_fpath = args.metadata_fpath
    reverb_fpath = args.reverb_fpath
    data_outdir = args.data_outdir
    mixture_type = eval(args.mixture_type)
    n_src = args.n_src

    start_time = time.time()
    create_mixtures(data_dir, noise_dir, dataset, metadata_fpath, reverb_fpath, data_outdir, n_src, mixture_type)
    end_time = time.time()
    print(f"Finish, took {end_time - start_time} seconds = {(end_time - start_time) / 60 / 60} hours")

def create_mixtures(data_dir, noise_dir, dataset, metadata_fpath, reverb_fpath, data_outdir, n_src, mixture_type):
    # Read metadata files
    md_df = pd.read_csv(metadata_fpath, engine='python')
    reverb_df = pd.read_csv(reverb_fpath, engine='python')
    assert len(md_df) == len(reverb_df), f"Mixture and reverb metadata file don't match"

    mode = Path(metadata_fpath).stem.split("_")[1]

    for i in range(len(mixture_type)):
        if mixture_type[i] in ['clean_sources', 'mix_sources', 'clean_sources_anechoic']:
            mixture_type_split = mixture_type[i].split("_", 1)
            mixture_type[i] = mixture_type_split[0] + f"_{n_src}{mixture_type_split[1]}"
        elif mixture_type[i] in ['reverb_noisy', 'anechoic_noisy']:
            mixture_type[i] = mixture_type[i] + f"_mix_{n_src}sources"
        else:
            raise NotImplementedError

    print(f"Generating mixtures for {metadata_fpath}")
    process_utterances(md_df, reverb_df, n_src, data_dir, noise_dir, data_outdir, mode, mixture_type, dataset)
    print(f"Done generating {metadata_fpath}")

def process_utterances(md_df, reverb_df, n_src, data_dir, noise_dir, data_outdir, mode, mixture_type, dataset):
    # Output directories
    root = os.path.join(data_outdir, f'{dataset}_mixture', mode)
    outpath_dict = {}
    for type in mixture_type:
        path_name = type
        path = os.path.join(root, type)
        outpath_dict[path_name] = path
        os.makedirs(path)
        print(f"Created path {path_name} = {path}")
    for i in range(n_src):
        if 'clean_2sources' in outpath_dict.keys():
            os.makedirs(os.path.join(outpath_dict['clean_2sources'], f's{i + 1}'))
        if 'clean_2sources_anechoic' in outpath_dict.keys():
            os.makedirs(os.path.join(outpath_dict['clean_2sources_anechoic'], f's{i + 1}'))

    # for results in tcc.process_map(
    #     functools.partial(
    #         process_single_utterance,
    #         data_dir, noise_dir, n_src),
    #     [row for _, row in md_df.iterrows()],
    #     chunksize=10,
    # ):
    for i in tqdm(range(len(md_df))):
        results = process_single_utterance(data_dir, noise_dir, n_src, md_df.loc[i])
        mixture_id, transformed_sources = results[0], results[1]
        if 'clean_2sources' in outpath_dict.keys():
            for i in range(n_src):
                sf.write(os.path.join(outpath_dict['clean_2sources'], f's{i+1}', mixture_id + '.wav'), transformed_sources[i], FREQUENCY)

        # Mix and write
        mixture = mix_sources(transformed_sources, n_src)
        if 'mix_2sources' in outpath_dict.keys():
            sf.write(os.path.join(outpath_dict['mix_2sources'], mixture_id + '.wav'), mixture, FREQUENCY)

        # Get RIRs
        if 'reverb_noisy_mix_2sources' in outpath_dict.keys() or 'anechoic_noisy_mix_2sources' in outpath_dict.keys() or 'clean_2sources_anechoic' in outpath_dict.keys():
            mixture_row = reverb_df[reverb_df['mixture_ID'] == mixture_id]
            room = WhamRoom([mixture_row['room_x'].iloc[0], mixture_row['room_y'].iloc[0], mixture_row['room_z'].iloc[0]],
                            [[mixture_row['micL_x'].iloc[0], mixture_row['micL_y'].iloc[0], mixture_row['mic_z'].iloc[0]],
                            [mixture_row['micR_x'].iloc[0], mixture_row['micR_y'].iloc[0], mixture_row['mic_z'].iloc[0]]],
                            [mixture_row['s1_x'].iloc[0], mixture_row['s1_y'].iloc[0], mixture_row['s1_z'].iloc[0]],
                            [mixture_row['s2_x'].iloc[0], mixture_row['s2_y'].iloc[0], mixture_row['s2_z'].iloc[0]],
                            mixture_row['T60'].iloc[0])
            room.generate_rirs()
            room.add_audio(transformed_sources[0], transformed_sources[1])
            anechoic = room.generate_audio(anechoic=True, fs=FREQUENCY)
            reverberant = room.generate_audio(fs=FREQUENCY)
            s1_spatial_scaling = np.sqrt(np.sum(transformed_sources[0] ** 2) / np.sum(anechoic[0, 0, :] ** 2))
            s2_spatial_scaling = np.sqrt(np.sum(transformed_sources[1] ** 2) / np.sum(anechoic[1, 0, :] ** 2))

        if 'clean_2sources_anechoic' in outpath_dict.keys():
            for i in range(n_src):
                sf.write(os.path.join(outpath_dict['clean_2sources_anechoic'], f's{i + 1}', mixture_id + '.wav'),
                         np.array(anechoic[i, 0, :len(mixture)]) * s1_spatial_scaling, FREQUENCY)


        if 'anechoic_noisy_mix_2sources' in outpath_dict.keys():
            anechoic_list = []
            anechoic_list.append(np.array(anechoic[0, 0, :len(mixture)]) * s1_spatial_scaling)
            anechoic_list.append(np.array(anechoic[1, 0, :len(mixture)]) * s2_spatial_scaling)
            anechoic_noisy_mixture = mix_sources(anechoic_list, n_src)
            anechoic_noisy_mixture = mix_noise(anechoic_noisy_mixture, transformed_sources[-1])
            sf.write(os.path.join(outpath_dict['anechoic_noisy_mix_2sources'], mixture_id + '.wav'), anechoic_noisy_mixture, FREQUENCY)

        if 'reverb_noisy_mix_2sources' in outpath_dict.keys():
            reverb_list = []
            reverb_list.append(np.array(reverberant[0, 0, :len(mixture)]) * s1_spatial_scaling)
            reverb_list.append(np.array(reverberant[1, 0, :len(mixture)]) * s2_spatial_scaling)
            reverb_noisy_mixture = mix_sources(reverb_list, n_src)
            reverb_noisy_mixture = mix_noise(reverb_noisy_mixture, transformed_sources[-1])
            reverb_noisy_mixture = check_for_clipping(reverb_noisy_mixture)
            sf.write(os.path.join(outpath_dict['reverb_noisy_mix_2sources'], mixture_id + '.wav'), reverb_noisy_mixture, FREQUENCY)

def process_single_utterance(wsj0_dir, noise_dir, n_src, row):
    # Get infos
    mixture_id, gain_list, sources_list, start_time_list, mixture_length = read_sources(row, wsj0_dir, noise_dir, n_src)
    # Transform sources according to the info
    transformed_sources = transform_sources(sources_list, gain_list, start_time_list, n_src, mixture_length)
    # Mix the sources
    return mixture_id, transformed_sources

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)