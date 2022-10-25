"""
This script is created to generate metadata file for NoiseX-92
"""

import argparse
import os
import pandas as pd
import soundfile as sf
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--noiseX92_dir', type=str, required=True,
                    help='Directory containing the noiseX-92 datasets')
parser.add_argument('--metadata_outpath', type=str, required=True,
                    help='Output path of the metadata file')

def main(args):
    path = args.noiseX92_dir
    metadata_outpath = args.metadata_outpath

    os.makedirs(metadata_outpath, exist_ok=True)

    metadata = pd.DataFrame(columns=['noise_ID', 'subset', 'length', 'augmented', 'origin_path'])

    noise_files = os.listdir(path)
    for file in noise_files:
        data, _ = sf.read(os.path.join(path, file))
        length = len(data)
        origin_path = os.path.join(Path(path).stem, file)
        metadata.loc[len(metadata)] = [file, 'tr', length, False, origin_path]

    metadata.to_csv(os.path.join(metadata_outpath, 'NoiseX-92_metadata.csv'), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

