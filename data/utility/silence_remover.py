import argparse
import os
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--metadata_dir', type=str, required=True,
                    help="Root directory to all the metadata files created")
parser.add_argument('--metadata_outdir', type=str, required=True,
                    help="Root directory to output the new metadata files created")

def main(args):
    metadata_dir = args.metadata_dir
    metadata_outdir = args.metadata_outdir

    metadata_files = os.listdir(metadata_dir)
    for metadata_file in metadata_files:
        print(f"Removing extra silence for {metadata_file}")
        md = pd.read_csv(os.path.join(metadata_dir, metadata_file))
        new_md = pd.DataFrame(columns = md.columns)
        for i in tqdm(range(len(md))):
            row = md.loc[i]
            s1_start = int(row['source_1_start'])
            s1_end = int(row['source_1_end'])
            s2_start = int(row['source_2_start'])
            s2_end = int(row['source_2_end'])
            mixture_length = int(row['audio_length'])
            mixture_ID = row['mixture_ID']
            source_1_path = row['source_1_path']
            source_1_speaker_ID = row['source_1_speaker_ID']
            source_1_gain = row['source_1_gain']
            source_2_path = row['source_2_path']
            source_2_speaker_ID = row['source_2_speaker_ID']
            source_2_gain = row['source_2_gain']
            noise_path = row['noise_path']
            noise_gain = row['noise_gain']
            noise_start = row['noise_start']

            # remove some silence if the silence on both sides is too long. we do not consider silence in the middle
            max_end = max([s1_end, s2_end])
            min_start = min([s1_start, s2_start])
            min_start_idx = [s1_start, s2_start].index(min_start)
            silence = min_start + (mixture_length - max_end)
            if silence > 0.1 * mixture_length:
                # remove 90% of this silence
                to_be_remove = int(min_start * 0.9)
                new_s1_start = s1_start - to_be_remove
                new_s1_end = s1_end - to_be_remove
                new_s2_start = s2_start - to_be_remove
                new_s2_end = s2_end - to_be_remove
                new_mixture_length = int(mixture_length - to_be_remove - 0.9 * (mixture_length - max_end))
                row = [mixture_ID, source_1_path, source_1_speaker_ID, source_1_gain, new_s1_start, new_s1_end,\
                       source_2_path, source_2_speaker_ID, source_2_gain, new_s2_start, new_s2_end, noise_path,\
                       noise_gain, noise_start, new_mixture_length]
                new_md.loc[len(new_md)] = row
        new_md.to_csv(os.path.join(metadata_outdir, metadata_file), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
