#!/usr/bin/python
# -*- coding:utf-8 -*-
from scipy.io import wavfile as wav
import numpy as np
from librosa import core
from librosa.feature import tempogram
from librosa.util.exceptions import ParameterError
import mir_eval
from glob import glob
from scipy import signal
import os

def read_tempofile(DB, genre, f):
    file_name = f.split('/')[-1].replace('wav', 'bpm')
    tempo_file = DB + '/key_tempo/' + genre + '/' + file_name
    with open(tempo_file, 'r') as f2:
        tempo = f2.read()
    return tempo

def read_beatfile(DB,f,genre=None, Dataset_name='Ballroom',LB=None):
    global reference_beats

    if Dataset_name == 'Ballroom':
        file_name = f.split('/')[-1].replace('wav', 'beats')
        beat_file = DB + '/key_beat/' + genre + '/' + file_name
        reference_beats, _ = mir_eval.io.load_labeled_events(beat_file)
        reference_beats = mir_eval.beat.trim_beats(reference_beats)
    elif Dataset_name == 'SMC':
        file_name = f.split('/')[-1].replace('.wav', '')
        beat_file = glob(LB + '/'+ file_name + '*.txt')[0]
        reference_beats = mir_eval.io.load_events(beat_file)
    elif Dataset_name == 'JCS':
        file_name = f.split('/')[-1].replace('.wav', '_beats.txt')
        beat_file = LB + '/' + file_name
        reference_beats, _ = mir_eval.io.load_labeled_events(beat_file)
        reference_beats = mir_eval.beat.trim_beats(reference_beats)
    return reference_beats

def read_downbeatfile(DB,f,genre=None,Dataset_name='Ballroom',LB=None):
    if Dataset_name == 'Ballroom':
        file_name = f.split('/')[-1].replace('wav', 'beats')
        beat_file = DB + '/key_beat/' + genre + '/' + file_name
        event_times, labels = mir_eval.io.load_labeled_events(beat_file)
    elif Dataset_name == 'JCS':
        file_name = f.split('/')[-1].replace('.wav', '_beats.txt')
        beat_file = LB + '/' + file_name
        #print("beat_file",beat_file)
        event_times, labels = mir_eval.io.load_labeled_events(beat_file)
    return event_times, labels

def read_wav(f):
    """Read wav audio and reformat type.
    Read in wav file and reformat the data type to 32-bit floating-point. And
    then, flatten to mono if it was stereo.
    Args:
        f: The audio filename.
    Returns:
        sr: Sampling rate of wav file.
        y: Data read from wav file.
    """
    sr, y = wav.read(f)

    if y.dtype == np.int16:
        y = y / 2 ** (16 - 1)
    elif y.dtype == np.int32:
        y = y / 2 ** (32 - 1)
    elif y.dtype == np.int8:
        y = (y - 2 ** (8 - 1)) / 2 ** (8 - 1)

    if y.ndim == 2:
        y = y.mean(axis=1)
    return (sr, y)

def P_score(t, gt):
    if abs((gt - t) / gt) <= 0.08:
        p = 1.0
    else:
        p = 0.0
    return p

def ALOTC(t_1, t_2, gt):
    if abs((gt - t_1) / gt) <= 0.08 or abs((gt - t_2) / gt) <= 0.08:
        p = 1.0
    else:
        p = 0.0
    return p

def tempo(y=None, sr=22050, onset_envelope=None, hop_length=512, start_bpm=120,
          std_bpm=1.0, ac_size=8.0, max_tempo=320.0, aggregate=np.mean):

    if start_bpm <= 0:
        raise ParameterError('start_bpm must be strictly positive')

    win_length = np.asscalar(core.time_to_frames(ac_size, sr=sr,
                                                 hop_length=hop_length))

    tg = tempogram(y=y, sr=sr,
                   onset_envelope=onset_envelope,
                   hop_length=hop_length,
                   win_length=win_length)

    # Eventually, we want this to work for time-varying tempo
    if aggregate is not None:
        tg = aggregate(tg, axis=1, keepdims=True)

    # Get the BPM values for each bin, skipping the 0-lag bin
    bpms = core.tempo_frequencies(tg.shape[0], hop_length=hop_length, sr=sr)

    # Weight the autocorrelation by a log-normal distribution
    prior = np.exp(-0.5 * ((np.log2(bpms) - np.log2(start_bpm)) / std_bpm)**2)

    prior2 = np.argsort(prior, axis=0)
    prior2_idx = prior2[-2]
    # print(prior2_idx)
    # print('prior_2_idx', prior2_idx)

    # Kill everything above the max tempo
    if max_tempo is not None:
        max_idx = np.argmax(bpms < max_tempo)
        prior[:max_idx] = 0

    # Really, instead of multiplying by the prior, we should set up a
    # probabilistic model for tempo and add log-probabilities.
    # This would give us a chance to recover from null signals and
    # rely on the prior.
    # it would also make time aggregation much more natural

    # Get the maximum, weighted by the prior

    period = tg * prior[:, np.newaxis]
    best_period = np.argmax(period, axis=0)
    best_2 = np.argsort(period, axis=0)
    prior2_idx = best_2[-2]
    #print(prior2_idx)
    #print(best_period)

    second_period = prior2_idx
    tempi = bpms[best_period]
    tempi2 = bpms[second_period]
    #print(type(tempi), type(tempi2))
    # Wherever the best tempo is index 0, return start_bpm
    tempi[best_period == 0] = start_bpm
    tempi2[second_period == 0] = start_bpm
    return (tempi2.astype(float)[0].item(), tempi.astype(float)[0].item())