import os
import sox
import argparse
import pandas as pd
from pathlib import Path
import subprocess
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--wsj_dir', type=str, required=True)
parser.add_argument('--wsj_splits', default='', type=str, required=False)
parser.add_argument('--data_outdir', type=str, required=True)
# parser.add_argument('--metadata_outdir', type=str, required=True)


def main(args):
    wsj_dir = args.wsj_dir
    wsj_splits = args.wsj_splits
    data_outdir = args.data_outdir
    # metadata_outdir = args.metadata_outdir

    os.makedirs(data_outdir, exist_ok=True)
    # os.makedirs(metadata_outdir, exist_ok=True)

    # metadata_train = pd.DataFrame(columns=['filename', 'speaker'])
    # metadata_dev = pd.DataFrame(columns=['filename', 'speaker'])
    # metadata_test = pd.DataFrame(columns=['filename', 'speaker'])

    wsj_dir_lst = []
    if wsj_splits:
        wsj_splits = wsj_splits.split(",")
        for split in wsj_splits:
            p = os.path.join(wsj_dir, split)
            wsj_dir_lst.append(p)

    else:
        for p in os.listdir(wsj_dir):
            wsj_dir_lst.append(p)

    # tfm = sox.Transformer()
    for dir in wsj_dir_lst:
        if "_tr_" in dir:
            mode = 'train'
        elif "_et_" in dir:
            mode = 'test'
        elif "_dt_" in dir:
            mode = 'dev'
        try:
            shutil.rmtree(f'{data_outdir}/temp')
        except:
            pass
        os.makedirs(f"{data_outdir}/temp")
        print(f"Getting wav files from {dir}, output to {data_outdir}/temp")
        subprocess.run(['bash', 'wsj0_mixture/conv.sh', f"{dir}", f"{data_outdir}/temp"])
        speakers = os.listdir(dir)
        for s in tqdm(speakers):
            files = os.listdir(os.path.join(dir, s))
            print(f"Creating output speaker specific directory: {os.path.join(data_outdir, mode, s)}")
            os.makedirs(os.path.join(data_outdir, mode, s), exist_ok=True)
            for f in files: # files with suffix .wv1 or .wv2
                f = f.split(".")[0] + ".wav"
                # eval(f"metadata_{mode}").loc[len(eval(f"metadata_{mode}"))] = [f"{mode}/{s}/{f}", s]
                subprocess.run(['cp', f"{data_outdir}/temp/{f}", f"{data_outdir}/{mode}/{s}"])
                # tfm.build(f"{dir}/{s}/{f}", f"{data_outdir}/{mode}/{s}/{f.split('.')[0]}.wav")

    # if not metadata_train.empty:
    #     metadata_train.to_csv(os.path.join(metadata_outdir, 'train_metadata.csv'), index=False)
    # if not metadata_test.empty:
    #     metadata_test.to_csv(os.path.join(metadata_outdir, 'test_metadata.csv'), index=False)
    # if not metadata_dev.empty:
    #     metadata_dev.to_csv(os.path.join(metadata_outdir, 'dev_metadata.csv'), index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
