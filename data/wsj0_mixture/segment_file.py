import argparse
import os
import pandas as pd
import sox
from tqdm import tqdm
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument('--wsj0_dir', type=str, required=True,
                    help="Root directory to wsj audios, it should contain all the subdirectory of INTERESTED_SUBSETS")
parser.add_argument('--wsj0_rttm_dir', type=str, required=True,
                    help="Directory to wsj0 rttm files")
parser.add_argument('--data_outdir', type=str, required=True,
                    help="Output directory for segmented audio files")
parser.add_argument('--metadata_outdir', type=str, required=True,
                    help="Putput directory for metadata file of segmented audio files")
parser.add_argument('--output_metadata', default=True,
                    help="Whether to output metadata file of segmented audio files")

INTERESTED_SUBSETS = ['train', 'test', 'dev']

def main(args):
    data_dir = args.wsj0_dir
    rttm_dir = args.wsj0_rttm_dir
    data_outdir = args.data_outdir
    metadata_outdir = args.metadata_outdir
    output_metadata = args.output_metadata

    rttm_files = os.listdir(rttm_dir)
    rttm_files = [file for file in rttm_files if
                  (file.endswith(".rttm")) & (file.split(".")[0] in INTERESTED_SUBSETS)]

    for file in rttm_files:
        dataset = file.split(".")[0]
        print(f"Segmenting for {dataset} ")
        md = pd.DataFrame(columns=['filename', 'origin_path', 'speaker', 'length'])
        with open(os.path.join(rttm_dir, file), 'r') as f:
            seg_num = 0
            last_file = ''
            f.readline()
            for line in tqdm(f):
                line = line.split(",")
                speaker = line[1]
                filename = line[0]
                if filename != last_file:
                    last_file = filename
                    seg_num = 0
                else:
                    seg_num += 1
                input = os.path.join(data_dir, dataset, speaker, filename + ".wav")
                start = float(line[2])
                end = start + float(line[3])
                root = os.path.join(data_outdir, dataset, speaker)
                os.makedirs(root, exist_ok=True)
                filename = f"{filename}-{seg_num}.wav"
                output = os.path.join(root, filename)
                tfm = sox.Transformer()
                tfm.trim(start, end)
                tfm.build_file(input, output)
                wav, _ = sf.read(output)
                md.loc[len(md)] = [filename, f"{dataset}/{speaker}/{filename}", speaker, len(wav)]
                seg_num += 1
        os.makedirs(metadata_outdir, exist_ok=True)
        if output_metadata:
            md.to_csv(os.path.join(metadata_outdir, f"{dataset}.csv"), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)