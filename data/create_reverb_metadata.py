"""
This script is created to add reverb to the mixtures generated.
Currently only supports mixture with 2 speakers.
"""

import argparse
import numpy as np
import pandas as pd
import os
import random
import glob
from pathlib import Path

from utility.sample_reverb import draw_params

random.seed(0)

MIC_SPACING = 0.05
REVERB_LEVEL='medium'


# Pass in arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mixture_md_dir', type=str, required=True,
                    help='Path to mixture metadata')
parser.add_argument('--metadata_outdir', type=str, required=True,
                    help='Path to store metadata for output mixture')

def main(args):
    mixture_md_dir = args.mixture_md_dir
    metadata_outdir = args.metadata_outdir

    mixture_md_list = glob.glob(f"{mixture_md_dir}/**/**.csv", recursive=True)
    mixture_md_list = [file for file in mixture_md_list if '2mixture_metadata' in file]

    for mixture_md in mixture_md_list:
        filename = Path(mixture_md).stem
        print(f"Sample reverb parameters for {filename}\n")
        mixture_md = pd.read_csv(mixture_md)

        utt_ids = mixture_md['mixture_ID']

        utt_list, param_list = [], []
        for utt in utt_ids:
            room_params = draw_params(MIC_SPACING, REVERB_LEVEL)
            room_dim = room_params[0]
            mics = room_params[1]
            s1 = room_params[2]
            s2 = room_params[3]
            T60 = room_params[4]

            param_dict = {'room_x': room_dim[0],
                          'room_y': room_dim[1],
                          'room_z': room_dim[2],
                          'micL_x': mics[0][0],
                          'micL_y': mics[0][1],
                          'micR_x': mics[1][0],
                          'micR_y': mics[1][1],
                          'mic_z': mics[0][2],
                          's1_x': s1[0],
                          's1_y': s1[1],
                          's1_z': s1[2],
                          's2_x': s2[0],
                          's2_y': s2[1],
                          's2_z': s2[2],
                          'T60': T60}

            utt_list.append(utt)
            param_list.append(param_dict)

        reverb_param_df = pd.DataFrame(data=param_list, index=utt_list,
                                       columns=['room_x', 'room_y', 'room_z', 'micL_x', 'micL_y', 'micR_x', 'micR_y', 'mic_z', 's1_x', 's1_y', 's1_z', 's2_x', 's2_y', 's2_z', 'T60'])

        reverb_param_path = os.path.join(metadata_outdir, filename.split(f'_metadata')[0] + '_reverb_params.csv')
        reverb_param_df.to_csv(reverb_param_path, index=True, index_label='mixture_ID')

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


