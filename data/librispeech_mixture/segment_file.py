import os
import sox
import argparse
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--librispeech_dir', type=str, required=True,
                    help="Root directory to librispeech audios, it should contain all the subdirectory of INTERESTED_SUBSETS")
parser.add_argument('--librispeech_rttm_dir', type=str, required=True,
                    help="Directory to librispeech rttm files")
parser.add_argument('--data_outdir', type=str, required=True,
                    help="Output directory for segmented audio files")
parser.add_argument('--metadata_outdir', type=str, required=True,
                    help="Putput directory for metadata file of segmented audio files")
parser.add_argument('--output_metadata', default=True,
                    help="Whether to output metadata file of segmented audio files")

# segment librispeech files
# generate metadata files should contain: filename, path, speaker, length

INTERESTED_SUBSETS = ['train-clean-360', 'train-clean-100', 'test-clean', 'dev-clean']

def main(args):
    data_path = args.librispeech_dir
    rttm_dir = args.librispeech_rttm_dir
    data_outdir = args.data_outdir
    metadata_outdir = args.metadata_outdir
    output_metadata = args.output_metadata
    
    rttm_files = os.listdir(rttm_dir)
    rttm_files = [file for file in rttm_files if (file.endswith(".rttm")) & (file.split(".")[0].replace("_", "-") in INTERESTED_SUBSETS)]


    for file in rttm_files:
        dataset = file.replace("_", "-").split(".")[0]
        print(f"Segmenting for {dataset} ")
        md = pd.DataFrame(columns = ['filename', 'origin_path', 'speaker', 'length'])
        with open(os.path.join(rttm_dir, file), 'r') as f:
            seg_num = 0
            last_file = ''
            for line in tqdm(f):
                line = line.split()
                speaker = line[7]
                book = line[1].split("-")[0]
                filename = f"{speaker}-{line[1]}"
                if filename != last_file:
                    last_file = filename
                    seg_num = 0
                else:
                    seg_num += 1
                input = os.path.join(data_path, dataset, speaker, book, filename + ".flac")
                start = float(line[3])
                end = start + float(line[4])
                root = os.path.join(data_outdir, dataset, speaker, book)
                os.makedirs(root, exist_ok=True)
                # file_path = f"{speaker}/{book}/{filename}-{seg_num}.wav"
                filename = f"{filename}-{seg_num}.wav"
                output = os.path.join(root, filename)
                tfm = sox.Transformer()
                tfm.trim(start, end)
                tfm.build_file(input, output)
                wav, _ = sf.read(output)
                md.loc[len(md)] = [filename, f"{dataset}/{speaker}/{book}/{filename}", speaker, len(wav)]
                seg_num += 1
        os.makedirs(metadata_outdir, exist_ok=True)
        if dataset in ['dev-clean', 'test-clean', 'train-clean-100']:
            dataset = dataset.split('-')[0]
        print(f"Saving file as {dataset}.csv")
        if output_metadata:
            md.to_csv(os.path.join(metadata_outdir, f"{dataset}.csv"), index=False)



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
