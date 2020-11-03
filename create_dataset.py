import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import tqdm
import ruamel.yaml

from preprocessing.text import Pipeline
from utils.audio import Audio

parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='CONFIG', type=str, required=True)
parser.add_argument('--dont_cache_phonemes', dest='CACHE_PHON', action='store_false')
parser.add_argument('--njobs', dest='NJOBS', type=int, default=16)
parser.add_argument('--col_sep', dest='COLUMN_SEP', type=str, default='|')
parser.add_argument('--recompute_phon', dest='RECOMPUTE_PHON', action='store_true')
args = parser.parse_args()
for arg in vars(args):
    print('{}: {}'.format(arg, getattr(args, arg)))
yaml = ruamel.yaml.YAML()
with open(str(Path(args.CONFIG) / 'data_config.yaml'), 'rb') as conf_yaml:
    config = yaml.load(conf_yaml)
args.DATA_DIR = config['data_directory']
args.META_FILE = os.path.join(args.DATA_DIR, config['metadata_filename'])
args.WAV_DIR = os.path.join(args.DATA_DIR, config['wav_subdir_name'])
args.TARGET_DIR = config['train_data_directory']
if args.TARGET_DIR is None:
    args.TARGET_DIR = args.DATA_DIR

mel_dir = os.path.join(args.TARGET_DIR, 'mels')
if not os.path.exists(mel_dir):
    os.makedirs(mel_dir)

phon_path = os.path.join(args.TARGET_DIR, 'phonemes.npy')
text_proc = Pipeline.default_training_pipeline(config['phoneme_language'], add_start_end=True)
# if os.path.exists(phon_path) and not args.RECOMPUTE_PHON:
#     print('Using cached phonemes.')
#     audio_data = np.load(phon_path)
# else:
print('\nLoading and cleaning text')

audio_data = []
with open(args.META_FILE, 'r', encoding='utf-8') as f:
    for l in f.readlines():
        l_split = l.split(args.COLUMN_SEP)
        filename, text = l_split[0], l_split[1]
        if filename.endswith('.wav'):
            filename = filename.split('.')[0]
        text = text_proc.cleaner(text)
        audio_data.append((filename, text))
audio_data = np.array(audio_data)
for i in range(len(audio_data)):
    audio_data[i][0] = audio_data[i][0].replace('\ufeff','')

print('\nPhonemizing')

texts = audio_data[:, 1]
batch_size = 250  # batch phonemization to avoid memory issues.
phonemes = []
for i in tqdm.tqdm(range(0, len(audio_data), batch_size)):
    batch = texts[i: i + batch_size]
    batch = text_proc.phonemizer(batch, njobs=args.NJOBS)
    phonemes.extend(batch)
audio_data = np.concatenate([np.array(audio_data), np.expand_dims(phonemes, axis=1)], axis=1)
if args.CACHE_PHON:
    np.save(phon_path, audio_data, allow_pickle=True)

print('\nBuilding dataset and writing files')
np.random.seed(42)
np.random.shuffle(audio_data)
test_metafile = os.path.join(args.TARGET_DIR, 'test_metafile.txt')
train_metafile = os.path.join(args.TARGET_DIR, 'train_metafile.txt')

test_lines = [''.join([filename, '|', text, '|', phon, '\n']) for filename, text, phon in
              audio_data[:config['n_test']]]
train_lines = [''.join([filename, '|', text, '|', phon, '\n']) for filename, text, phon in
               audio_data[config['n_test']:-1]]

with open(test_metafile, 'w+', encoding='utf-8') as test_f:
    test_f.writelines(test_lines)
with open(train_metafile, 'w+', encoding='utf-8') as train_f:
    train_f.writelines(train_lines)

audio = Audio(config)
for i in tqdm.tqdm(range(len(audio_data))):
    filename, _, _ = audio_data[i]
    wav_path = os.path.join(args.WAV_DIR, filename + '.wav')
    y, sr = librosa.load(wav_path, sr=config['sampling_rate'])
    mel = audio.mel_spectrogram(y)
    mel_path = os.path.join(mel_dir, filename)
    np.save(mel_path, mel.T)
print('\nDone')
