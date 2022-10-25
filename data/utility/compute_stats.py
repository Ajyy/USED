import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
#import torch
import time

parser = argparse.ArgumentParser()
parser.add_argument('--metadata_dir', type=str, required=True,
                    help='Path to the metadata directory')

RATE = 16000

def compute_total(args):
    metadata_dir = args.metadata_dir

    metadata_files = os.listdir(metadata_dir)
    metadata_files = [file for file in metadata_files if ('mixture_metadata' in file and file.endswith('csv'))]

    total_hour = 0
    for file in metadata_files:
        md_file = pd.read_csv(os.path.join(metadata_dir, file))
        total_length = md_file['audio_length']
        hour = sum(total_length) / RATE / 60 / 60
        print(f"{file} has {hour} hours\n")
        total_hour += hour
    print(f"total_hour = {total_hour}\n")

def compute_by_scenarios(args):
    metadata_dir = args.metadata_dir

    metadata_files = os.listdir(metadata_dir)
    metadata_files = [file for file in metadata_files if ('mixture_metadata' in file and file.endswith('csv'))]
    print(metadata_files)


    for file in metadata_files:
        print(f"Computing hours for {file}")
        md_file = pd.read_csv(os.path.join(metadata_dir, file))
        case_breakdown = md_file.apply(lambda row: define_case(row), axis=1)

        qq_total = 0
        sq_total = 0
        ss_total = 0
        qs_total = 0
        total_length = 0
        for i in range(len(case_breakdown)):
            case_dict = case_breakdown.loc[i]
            assert all(val >= 0 for val in list(case_dict.values())), f"Check values = {case_dict}, some values are negative, from row = {md_file.loc[i]['mixture_ID']}"
            qq_total += case_dict['qq'] / RATE / 60 / 60
            sq_total += case_dict['sq'] / RATE / 60 / 60
            ss_total += case_dict['ss'] / RATE / 60 / 60
            qs_total += case_dict['qs'] / RATE / 60 / 60
            total_length += md_file.loc[0]['audio_length']
        print(f"{file} has qq = {qq_total}, sq = {sq_total}, ss = {ss_total}, qs = {qs_total}\n")

def define_case(row):
    s1_start = int(row['source_1_start'])
    s1_end = int(row['source_1_end'])
    s2_start = int(row['source_2_start'])
    s2_end = int(row['source_2_end'])
    total_length = int(row['audio_length'])
    if (s2_start >= s1_start and s2_end >= s1_end and s2_start <= s1_end):
        # print("case 1")
        return {'qq': s1_start + total_length - s2_end,
                'sq': s2_start - s1_start,
                'ss': s1_end - s2_start,
                'qs': s2_end - s1_end}
    elif (s2_start >= s1_start and s2_end <= s1_end):
        # print("case 2")
        return {'qq': s1_start + total_length - s1_end,
                'sq': s2_start - s1_start + s1_end - s2_end,
                'ss': s2_end - s2_start,
                'qs': 0}
    elif (s2_start >= s1_end):
        # print("case 3")
        return {'qq': s1_start + s2_start - s1_end + total_length - s2_end,
                'sq': s1_end - s1_start,
                'ss': 0,
                'qs': s2_end - s2_start}
    elif (s1_start >= s2_start and s1_end >= s2_end and s1_start <= s2_end):
        # print("case 4")
        return {'qq': s2_start + total_length - s1_end,
                'qs': s1_start - s2_start,
                'ss': s2_end - s1_start,
                'sq': s1_end - s2_end}
    elif (s1_start >= s2_start and s1_end <= s2_end):
        # print("case 5")
        return {'qq': s2_start + total_length - s2_end,
                'qs': s1_start - s2_start + s2_end - s1_end,
                'ss': s1_end - s1_start,
                'sq': 0}
    elif (s1_start >= s2_end):
        # print("case 6")
        return {'qq': s2_start + s1_start - s2_end + total_length - s1_end,
                'qs': s2_end - s2_start,
                'ss': 0,
                'sq': s1_end - s1_start}
    else:
        print("This case is not considered:")
        print(s1_start, s1_end, s2_start, s2_end, total_length)
        print("Please implement")
        raise
# def get_labels(args):
#     metadata_dir = args.metadata_dir
#
#     metadata_files = os.listdir(metadata_dir)
#     metadata_files = [file for file in metadata_files if ('mixture_metadata' in file and file.endswith('csv'))]
#     print(metadata_files)
#
#     for file in metadata_files:
#         print(f"Computing for {file}")
#         case_dict = {'qq': 0, 'sq': 0, 'ss': 0, 'qs': 0}
#         md_file = pd.read_csv(os.path.join(metadata_dir, file))
#         start_time = time.time()
#         for i in tqdm(range(len(md_file))):
#             s1_start = md_file.loc[i, 'source_1_start']
#             s1_end = md_file.loc[i, 'source_1_end']
#             s2_start = md_file.loc[i, 'source_2_start']
#             s2_end = md_file.loc[i, 'source_2_end']
#             mixture_length = md_file.loc[i, 'audio_length']
#             label_tgt = np.zeros(mixture_length)
#             label_tgt[int(s1_start): int(s1_end)] = 1
#             label_int = np.zeros(mixture_length)
#             label_int[int(s2_start): int(s2_end)] = 1
#
#             qq, sq, ss, qs = get_cases_length(label_tgt, label_int)
#             assert (qq + sq + ss + qs) == mixture_length
#             case_dict['qq'] += qq / 16000 / 60 / 60
#             case_dict['sq'] += sq / 16000 / 60 / 60
#             case_dict['ss'] += ss / 16000 / 60 / 60
#             case_dict['qs'] += qs / 16000 / 60 / 60
#             # case_dict['qq'] += qq
#             # case_dict['sq'] += sq
#             # case_dict['ss'] += ss
#             # case_dict['qs'] += qs
#         end_time = time.time()
#         case_dict['qq'] = case_dict['qq']
#         case_dict['sq'] = case_dict['sq']
#         case_dict['ss'] = case_dict['ss']
#         case_dict['qs'] = case_dict['qs']
#         print(f"Case summary = {case_dict}. Took {end_time - start_time} seconds")
#         exit()
#
# def get_cases_length(label_tgt, label_int):
#     # label_tgt = torch.from_numpy(label_tgt)
#     # label_int = torch.from_numpy(label_int)
#     qq = sum(((label_tgt == label_int) & (label_tgt == 0)))
#     sq = sum(((label_tgt != label_int) & (label_tgt == 1)))
#     ss = sum(((label_tgt == label_int) & (label_tgt == 1)))
#     qs = sum(((label_tgt != label_int) & (label_tgt == 0)))
#     return qq, sq, ss, qs


if __name__ == "__main__":
    args = parser.parse_args()
    compute_total(args)
    compute_by_scenarios(args)
    # get_labels(args) # this method is very slow

