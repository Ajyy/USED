import argparse
import os
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import glob

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--md_path', type=str, required=True,
                    help="Path to  in specific metadata file")
parser.add_argument('--dataset_name', type=str, required=True,
                    help="Either wsj0 or LibriSpeech")
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--num_ref', type=int, default=1, required=True,
                    help='Number of references for each mixture')
parser.add_argument('--ref_outdir', type=str, required=True)


def librispeech_ref(args):
    md_path = args.md_path
    dataset_name = args.dataset_name
    data_dir = args.data_dir
    num_ref = args.num_ref
    outdir = args.ref_outdir

    ref_md = pd.DataFrame(columns=['mixture_ID'])
    for num in range(num_ref):
        ref_md[f'mixture_ref_{num + 1}'] = {}


    md = pd.read_csv(md_path, engine='python')
    for i in tqdm(range(len(md))):
        source_1_path = md.loc[i]['source_1_path'].split("/")
        source_1_speaker = md.loc[i]['source_1_speaker_ID']
        mixture_ID = md.loc[i]['mixture_ID']
        source_1_audio_name = source_1_path[-1]
        mode = source_1_path[0]
        try:
            if isinstance(eval(source_1_speaker), float): # speakers in test set have no characters and is intepret as float
                source_1_speaker = str(int(eval(source_1_speaker)))
        except ValueError:
            source_1_speaker = str(int(source_1_speaker))
        possible = glob.glob(f"{data_dir}/{mode}/{source_1_speaker}/*/*.wav")
        # possible = os.listdir(os.path.join(data_dir, mode, source_1_speaker))
        possible = [audio for audio in possible if source_1_audio_name not in audio]
        choices = random.sample(possible, num_ref)
        row = [mixture_ID]
        for choice in choices:
            # ref_path = os.path.join(data_dir, mode, source_1_speaker, choice)
            assert os.path.exists(choice), f"Path = {ref_path} not exist"
            ref_path = "/".join(choice.split("/")[-3:])
            row.append(os.path.join(mode, ref_path))
        ref_md.loc[len(ref_md)] = row

    mode = Path(md_path).stem.split("_")[1]
    out_path = os.path.join(outdir, f'{dataset_name}_{mode}_2mixture_ref.csv')
    ref_md.to_csv(out_path, index=False)



def wsj0_ref(args):
    md_path = args.md_path
    dataset_name = args.dataset_name
    data_dir = args.data_dir
    num_ref = args.num_ref
    outdir = args.ref_outdir

    ref_md = pd.DataFrame(columns=['mixture_ID'])
    for num in range(num_ref):
        ref_md[f'mixture_ref_{num + 1}'] = {}


    md = pd.read_csv(md_path, engine='python')
    for i in tqdm(range(len(md))):
        source_1_path = md.loc[i]['source_1_path'].split("/")
        source_1_speaker = md.loc[i]['source_1_speaker_ID']
        mixture_ID = md.loc[i]['mixture_ID']
        source_1_audio_name = source_1_path[-1]
        mode = source_1_path[0]
        try:
            possible = glob.glob(f"{data_dir}/{mode}/{int(float(source_1_speaker))}/*.wav")
        except:
            possible = []
        if not possible:
            possible = glob.glob(f"{data_dir}/{mode}/{source_1_speaker}/*.wav")
        # possible = os.listdir(os.path.join(data_dir, mode, source_1_speaker))
        possible = [audio for audio in possible if source_1_audio_name not in audio]
        choices = random.sample(possible, num_ref)
        row = [mixture_ID]
        for choice in choices:
            # ref_path = os.path.join(data_dir, mode, source_1_speaker, choice)
            assert os.path.exists(choice), f"Path = {ref_path} not exist"
            ref_path = "/".join(choice.split("/")[-3:])
            # row.append(os.path.join(mode, ref_path))
            row.append(ref_path)
        ref_md.loc[len(ref_md)] = row

    mode = Path(md_path).stem.split("_")[1]
    out_path = os.path.join(outdir, f'{dataset_name}_{mode}_2mixture_ref.csv')
    ref_md.to_csv(out_path, index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.dataset_name == 'LibriSpeech':
        librispeech_ref(args)
    elif args.dataset_name == 'wsj0':
        wsj0_ref(args)
    else:
        raise NotImplementedError


