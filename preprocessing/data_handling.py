import os
from random import Random

import numpy as np
import tensorflow as tf

from preprocessing.text.tokenizer import Tokenizer


class Dataset:
    """ Model digestible dataset. """
    
    def __init__(self,
                 samples,
                 preprocessor,
                 batch_size,
                 shuffle=True,
                 drop_remainder=True,
                 mel_channels=80,
                 seed=42):
        self._random = Random(seed)
        self._samples = samples[:]
        self.preprocessor = preprocessor
        output_types = (tf.float32, tf.int32, tf.int32)
        padded_shapes = ([-1, mel_channels], [-1], [-1])
        dataset = tf.data.Dataset.from_generator(lambda: self._datagen(shuffle),
                                                 output_types=output_types)
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes=padded_shapes,
                                       drop_remainder=drop_remainder)
        self.dataset = dataset
        self.data_iter = iter(dataset.repeat(-1))
    
    def next_batch(self):
        return next(self.data_iter)
    
    def all_batches(self):
        return iter(self.dataset)
    
    def _datagen(self, shuffle):
        """
        Shuffle once before generating to avoid buffering
        """
        samples = self._samples[:]
        if shuffle:
            # print(f'shuffling files')
            self._random.shuffle(samples)
        return (self.preprocessor(s) for s in samples)


def load_files(metafile,
               meldir,
               num_samples=None):
    samples = []
    count = 0
    alphabet = set()
    with open(metafile, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            l_split = l.split('|')
            mel_file = os.path.join(str(meldir), l_split[0] + '.npy')
            text = l_split[1].strip().lower()
            phonemes = l_split[2].strip()
            samples.append((phonemes, text, mel_file))
            alphabet.update(list(text))
            count += 1
            if num_samples is not None and count > num_samples:
                break
        alphabet = sorted(list(alphabet))
        return samples, alphabet


class DataPrepper:
    
    def __init__(self,
                 config,
                 tokenizer: Tokenizer):
        self.start_vec = np.ones((1, config['mel_channels'])) * config['mel_start_value']
        self.end_vec = np.ones((1, config['mel_channels'])) * config['mel_end_value']
        self.tokenizer = tokenizer
    
    def __call__(self, sample):
        phonemes, text, mel_path = sample
        mel = np.load(mel_path)
        return self._run(phonemes, text, mel)
    
    def _run(self, phonemes, text, mel):
        encoded_phonemes = self.tokenizer(phonemes)
        norm_mel = np.concatenate([self.start_vec, mel, self.end_vec], axis=0)
        stop_probs = np.ones((norm_mel.shape[0]))
        stop_probs[-1] = 2
        return norm_mel, encoded_phonemes, stop_probs


class ForwardDataPrepper:
    
    def __call__(self, sample):
        mel, encoded_phonemes, durations = np.load(str(sample), allow_pickle=True)
        return mel, encoded_phonemes, durations
