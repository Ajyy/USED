import argparse
import os
import numpy as np
from tqdm import tqdm


def Vad(audio_file_path):
    from speechbrain.pretrained import VAD
    # print(audio_file_path)
    VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")

    # 1- Let's compute frame-level posteriors first
    prob_chunks = VAD.get_speech_prob_file(audio_file_path)

    # 2- Let's apply a threshold on top of the posteriors
    prob_th = VAD.apply_threshold(prob_chunks).float()

    # 3- Let's now derive the candidate speech segments
    boundaries = VAD.get_boundaries(prob_th)

    # for some wav, there may be error
    # 4
    try:
        boundaries = VAD.energy_VAD(audio_file_path, boundaries)
    except Exception:
        print('error:', audio_file_path)

    # 5- Merge segments that are too close
    boundaries = VAD.merge_close_segments(boundaries, close_th=0.2)

    # 6- Remove segments that are too short
    boundaries = VAD.remove_short_segments(boundaries, len_th=0.20)

    # print(boundaries)

    boundaries = VAD.double_check_speech_segments(boundaries, audio_file_path, speech_th=0.6)
    # VAD.save_boundaries(boundaries)

    return boundaries


if __name__ == '__main__':

    parser = argparse.ArgumentParser('SP_ID')

    # training
    parser.add_argument('--wsj_dir', type=str,
                        help='the directory containing train dev and test folders; The sample rate of audio is 16k and format is changed to wav')
    parser.add_argument('--metadata_outdir', type=str,
                        help='output folder that saves vad_output infomation')

    args = parser.parse_args()

    input_dir = args.metadata_outdir
    output_folder = args.data_outdir

    kinds = ['test', 'dev', 'train']

    for kind in kinds:
        path = os.path.join(input_dir, kind)

        speech_boundary_list = []

        for speaker in tqdm(os.listdir(path)):
            speaker_path = os.path.join(path, speaker)
            for utt in os.listdir(speaker_path):
                utt_id = utt.split('.')[0]
                utt_path = os.path.join(speaker_path, utt)
                speech_boundaries = Vad(utt_path).numpy()

                for x in speech_boundaries:
                    speech_boundary_list.append([utt_id, speaker, round(x[0], 2), round(x[1] - x[0], 2)])
        os.makedirs(output_folder, exist_ok=True)

        with open(os.path.join(output_folder, str(kind) + '.rttm'), 'w') as p:
            p.write('utterance_id,speaker_id,start_time,length\n')
            for line in speech_boundary_list:
                p.write(str(line[0]) + ',' + str(line[1]) + ',' + str(line[2]) + ',' + str(line[3]) + '\n')
