"""
proposal: Have you chosen a correct background music?
"""
import os
import librosa
import pandas as pd
import matplotlib.pyplot as plt


def load_class_info(path):
    if not os.path.exists(path):
        create_class_info(path)
    return pd.read_csv(path)


def create_class_info(path):
    with open(path, "w") as info_file:
        info_file.write("audio_name,audio_path,mood\n")
        for c in _classes:
            for audio in os.listdir(f"{_audio_source_path}{c}/"):
                info_file.write(f"{audio},{_audio_source_path}{c}/{audio},{c}\n")


def generate_spectrogram():
    pass


def generate_wavelet():
    pass


def run():
    pass


_classes = ["aggressive", "dramatic", "happy", "romantic", "sad"]
_audio_source_path = "music_mood/"
_data_source_path = "ProcessedData/"
_audio_class_source = f"{_data_source_path}baseInfo.csv"
_audio_class_info = load_class_info(_audio_class_source)
_sample_rate = 22050


for audio in _audio_class_info.iloc[:,1]:
    signal, sample_rate = librosa.load(audio, sr=_sample_rate)
    print(sample_rate)

