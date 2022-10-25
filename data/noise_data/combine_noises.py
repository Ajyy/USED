"""
This script is created to combine different noise datasets.
Currently, it combines wham noises and noisex-92 metadata files.
You can edit this script to handle more noise datasets.
"""

import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--noiseX_92_md_path', type=str, required=True,
                    help='Path noisex-92 metadata')
# parser.add_argument('--wham_dir', type=str, required=True,
#                     help='Directory containing all the wham noise for different wham datasets')
parser.add_argument('--wham_md_dir', type=str, required=True,
                    help='Directory containing all the wham noise metadata files for different wham datasets')
parser.add_argument('--metadata_outdir', type=str, required=True,
                    help='Directory for outputing different metadata')


def main(args):
    noiseX_92_md_path = args.noiseX_92_md_path
    wham_md_dir = args.wham_md_dir
    metadata_outdir = args.metadata_outdir

    noiseX_92_md = pd.read_csv(noiseX_92_md_path, engine='python')

    os.makedirs(metadata_outdir, exist_ok=True)

    wham_md_lst = os.listdir(wham_md_dir)
    for wham_md in wham_md_lst:
        print(f"Combining metadata with {wham_md} & Noise_X-92")
        outpath = os.path.join(metadata_outdir, 'all_noises_' + wham_md)
        wham_md = pd.read_csv(os.path.join(wham_md_dir, wham_md), engine='python')
        for i in range(len(wham_md)):
            wham_md.iat[i, wham_md.columns.get_loc('origin_path')] = os.path.join('wham_noise', wham_md.iloc[i]['origin_path'])
        result = pd.concat([noiseX_92_md, wham_md])
        result = result.sort_values(by='length')
        result.to_csv(outpath, index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
