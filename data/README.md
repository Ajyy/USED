# A. Prepare Noise Data

## Step1: Get Noise Data & Create Metadata
Get noises data: 
1. wham noise: https://github.com/s3prl/LibriMix 
   1. In this github repository, you can 'generate_librimix_sd.sh' to download needed data but comment out unnecessary parts
   2. You can download these datasets using this script: LibriSpeech_dev_clean, LibriSpeech_test_clean, LibriSpeech_clean100, Librispeech_clean360 (All these data are needed for data preparation later. Please download all.)
2. NoiseX-92: https://github.com/speechdnn/Noises/tree/master/NoiseX-92 

From LibriMix repository, you will also get all the metadata files. 
The metadata files provided include:
1. LibriMix/metadata/Wham_noise 
2. LibriMix/metadata/LibriSpeech
3. Others are not needed in our data preparation.


Get metadata for NoiseX-92:
```sh
python noise_data/create_noisex-92_metadata.py \
    --noiseX92_dir {YOUR_DIRECTORY/NoiseX-92} \
    --metadata_outpath {NoiseX-92_metadata_output_path}
```
This will create a file named 'NoiseX-92_metadata.csv' in you {NoiseX-92_metadata_output_path}

## Step1.1 (Optioal): Augment Wham Noise
Follow this script from LibriMix respository: 
https://github.com/s3prl/LibriMix/blob/master/scripts/augment_train_noise.py

For this project, we will be using augmented wham noise data. You can skip this step if you intend to generate smaller dataset.

## Step2: Combine All the Noise Data & Get Metadata
```sh
$ python noise_data/combine_noises.py \
    --noiseX_92_md_path {YOUR_DIRECTORY/NoiseX-92_metadata.csv} \
    --wham_md_dir {YOUR_DIRECTORY/LibriMix/metadata/Wham_noise} \
    --metadata_outdir {Directory_for_noise_metadata}
```
This will create three csv files:
1. all_noises_dev.csv
2. all_noises_test.csv
3. all_noises_train.csv

Move all the noises data to the same folder. Here is the structure of nosie folder:
```sh
├── NoiseX-92
│   ├── babble.wav
│   ├── buccaneer1.wav
│   ├── buccaneer2.wav
│   ├── ...
│   └── white.wav
└── wham_noise
    ├── cv
    ├── tr
    └── tt
```
This directory will be called {YOUR_DIRECTORY_TO_ALL_NOISE} in later sections.




# B. Pre-Mixture-Generating Steps 

# B1. LibriSpeech

## Step1: Prepare LibriSpeech Data
LibriSpeech dataset can be downloaded from: see Section A step1.


## Step2(*): Get LibriSpeech Metadata Files
LibriSpeech metadata files can be downloaded from: see Section A step1.
You will need these rttm files for Step 3 in this section:
1. dev_clean.rttm
2. test_clean.rttm
3. train_clean_100.rttm
4. train_clean_360.rttm

## Step3: Get Segmented Audios & Metadata File
```shell
python librispeech_mixture/segment_file.py \
    --librispeech_dir {Directory_to_all_LibriSpeech_audios} \
    --librispeech_rttm_dir {Directory_to_all_librispeech_rttm_files} \
    --data_outdir {Directory_for_segmented_audios_ls} \
    --metadata_outdir {Directory_for_metadata_of_segmented_audios_ls} \
    --output_metadata False
```
Change to "--output_metadata True" if you intend to perform Step3 below. And you should get the following metadata files:
1. dev.csv (from dev-clean.csv)
2. test.csv (from test-clean.csv)
3. train.csv (from train-clean-100.csv)
4. train-clean-360.csv

<!--
# B2. DIHARD

## Step1: Prepare DIHARD II Data
Get license and download from LDC


## Step2: Get Segmented Audios & Metadata File
DIHARD are conversation data. To generate our own dataset, we segment the data first to get audios that contains only a single spearker.
According to the rttm files provided in the dataset, we can get the single-speaker audio segments.
```sh
python dihard_mixture/segment_file.py \
    --dihard_dir {YOUR_DIRECTORY_TO_DIHARD/data/single_channel} \
    --data_outdir {Directory_for_segmented_data_dh} \
    --metadata_outdir {Directory_for_metadata_of_segmented_audios_dh} \
    --output_metadata False
```
Change to "--output_metadata True" if you intend to perform Step3 below. This will generate the following metadata file:
dihard_all_segmented_files.csv
-->

# B2. WSJ0

## Step1: Prepare WSJ0
Get licence and download from:?

## Step2: Convert Data & Generate Metadata
We only use si_tr_s (training set), si_dt_05 (development set), si_et_05 (test set) to generate data.

Convert downloaded data to .wav format & generate metadata, the sampling rate is 16k:
```sh
python wsj0_mixture/conv.py \
    --wsj_dir {Directory_to_wsj0_data} \
    --wsj_splits si_tr_s,si_et_05,si_dt_05 \
    --data_outdir {Directory_for_converted_wav}
```

## Step3: Get Segmented Infomation
Since there is not open source rttm like file for wsj0 to indicate the start and end time of voice in each utterance, 
we generate this metadata file on our own. 
This script only generates SPEECH segment info.
```sh
python wsj0_mixture/generate_segement_vad.py \
    --wsj_dir {Directory_for_converted_wav}\
    --metadata_outdir {Directory_for_rttm_file_wsj0}
```

## Step4: Get Segmented Audios & Metadata File
```shell
python wsj0_mixture/segment_file.py \
  --wsj0_dir {Directory_for_converted_wav} \
  --wsj0_rttm_dir {Directory_for_rttm_file_wsj0} \
  --data_outdir {Directory_for_segmented_data_wsj0} \
  --metadata_outdir {Directory_for_metadata_of_segmented_data_wsj} \
  --output_metadata False
```



# C. Create Mixture Metadata from Different Datasets
You don't need to go through this step if you use the metadata files we provide.
```shell
python create_mixture_metadata.py \
   --dataset_name {wsj0 or LibriSpeech} \
   --data_dir {Directory_for_segmented_data_wsj0_or_librispeech} \
   --data_md_fpath {Directory_to_metadata_files/train.csv or test.csv or dev.csv or train-clean-360.csv} \
   --noise_dir {Directory_to_all_noise_data} \
   --noise_md_dir {Directory_for_noise_metadata} \
   --metadata_outdir {Directory_for_mixture_metadata} \
   --output_num {Number_of_utterances_to_be_generated}
```
In our script, we will generate two separate sets of train and dev:
LibriSpeech_dev-clean-360_2mixture_metadata.csv & LibriSpeech_train-clean-360_2mixture_metadata.csv
LibriSpeech_dev_2mixture_metadata.csv & LibriSpeech_train_2mixture_metadata.csv

# D. Sample Reverb Parameters for All Mixture Metadata
You don't need to go through this step if you use the metadata files we provide.
```sh
python utility/create_reverb_metadata.py \
    --mixture_md_dir {Output_directory_for_mixture_metadata} \
    --metadata_outdir {Output_directory_for_reverb_parameters_metadata}
```



# E. Create Actual Mixtures
Note that all the audios are set to frequency=16k.

```shell
python create_mixture.py \
  --data_dir {Directory_to_seg,emted_data_wsj0_or_librispeech} \
  --dataset {wsj0 or LibriSpeech} \
  --noise_dir {Directory_to_all_noise_data} \
  --metadata_fpath {Path_of_mixture_metadata_file/**mixture_metadata.csv} \
  --reverb_fpath {Path_of_reverb_metadata_file/**_reverb_params.csv} \
  --data_outdir {Output_directory_for_generated_data}
```

# Update:
There are no documentation for files "create_mixture_metadata_ss_only.py" & "create_mixture_ss_only.py", and "utility/create_mixture_helper_ss_only.py" & "generate_metadata_helper_ss_only.py" because they follow the same logic as files with same name but without suffix "_ss_only".
