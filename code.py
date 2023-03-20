"""
proposal: Have you chosen a correct background music?
"""
import os
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from tqdm import tqdm


def load_class_info():
    if not os.path.exists(_audio_class_source):
        create_class_info()
    return pd.read_csv(_audio_class_source)


def create_class_info():
    with open(_audio_class_source, "w") as info_file:
        info_file.write("audio_name,audio_path,mood\n")
        for c in _moods:
            for audio in os.listdir(f"{_audio_source_path}{c}/"):
                info_file.write(f"{audio},{_audio_source_path}{c}/{audio},{c}\n")


def load_mfcc():
    if not os.path.exists(_audio_mfcc_source):
        create_mfcc()
    with open(_audio_mfcc_source, "r") as mfcc_file:
        data = json.load(mfcc_file)
        mfccs = np.array(data["mfcc"])
        moods = np.array(list(map(_moods.index, data["mood"])))
    print(f"{get_mfcc_source()} Loaded...")
    return mfccs, moods


def create_mfcc():
    data = {
        "mfcc": [],
        "mood": []
    }
    with tqdm(total=_total, desc="Processing MFCC") as pbar:
        for audio, path, label in _audio_class_info.itertuples(index=False):
            signal, sample_rate = librosa.load(path, sr=_sample_rate)
            mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=_num_fft, n_mfcc=_num_mfcc, hop_length=_hop_length)
            data["mfcc"].append(mfcc.tolist())
            data["mood"].append(label)
            pbar.update(1)
    
    with open(_audio_mfcc_source, "w") as mfcc_file:
        json.dump(data, mfcc_file, indent=2)
    print(f"{get_mfcc_source()} Created...")
    


def generate_spectrogram():
    pass


def generate_wavelet():
    pass


def get_mfcc_source():
    return f"{_data_source_path}mfcc_{_num_mfcc}_{_num_fft}_{_hop_length}.json"


_moods = ["aggressive", "dramatic", "happy", "romantic", "sad"]
_audio_source_path = "music_mood/"
_data_source_path = "ProcessedData/"
_audio_class_source = f"{_data_source_path}baseInfo.csv"
_audio_class_info = load_class_info()
_total = 2500

_sample_rate = 44100
_hop_length = 512
_num_fft = 2048
_num_mfcc = 20
_audio_mfcc_source = get_mfcc_source()


def run(hop_length, n_fft, n_mfcc):
    _hop_length = hop_length
    _num_fft = n_fft
    _num_mfcc = n_mfcc
    _audio_mfcc_source = get_mfcc_source()
    
    X, y = load_mfcc()

X, y = load_mfcc()
print(X)
