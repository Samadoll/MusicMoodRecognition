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

class Music_Model:
    
    def __init__(self, hop_length=512, num_fft=2048, num_mfcc=20):
        # Default Values
        self._moods = ["aggressive", "dramatic", "happy", "romantic", "sad"]
        self._audio_source_path = "music_mood/"
        self._data_source_path = "ProcessedData/"
        self._audio_class_source = f"{self._data_source_path}baseInfo.csv"
        self._total = 2500
        self._sample_rate = 44100

        # Variables
        self._hop_length = hop_length
        self._num_fft = num_fft
        self._num_mfcc = num_mfcc
        self._audio_mfcc_source = self.get_mfcc_source()
        self._audio_class_info = self.load_class_info()
        


    def load_class_info(self):
        if not os.path.exists(self._audio_class_source):
            self.create_class_info()
        return pd.read_csv(self._audio_class_source)


    def create_class_info(self):
        with open(self._audio_class_source, "w") as info_file:
            info_file.write("audio_name,audio_path,mood\n")
            for c in self._moods:
                for audio in os.listdir(f"{self._audio_source_path}{c}/"):
                    info_file.write(f"{audio},{self._audio_source_path}{c}/{audio},{c}\n")
        print(f"{self._audio_class_source} Created...")


    def load_mfcc(self):
        if not os.path.exists(self._audio_mfcc_source):
            self.create_mfcc()
        with open(self._audio_mfcc_source, "r") as mfcc_file:
            data = json.load(mfcc_file)
            mfccs = np.array(data["mfcc"])
            moods = np.array(list(map(self._moods.index, data["mood"])))
        print(f"{self.get_mfcc_source()} Loaded...")
        return mfccs, moods


    def create_mfcc(self):
        data = {
            "mfcc": [],
            "mood": []
        }
        with tqdm(total=self._total, desc="Processing MFCC") as pbar:
            for audio, path, label in self._audio_class_info.itertuples(index=False):
                signal, sample_rate = librosa.load(path, sr=self._sample_rate)
                mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=self._num_fft, n_mfcc=self._num_mfcc, hop_length=self._hop_length)
                data["mfcc"].append(mfcc.tolist())
                data["mood"].append(label)
                pbar.update(1)
        
        with open(self._audio_mfcc_source, "w") as mfcc_file:
            json.dump(data, mfcc_file, indent=2)
        print(f"{self.get_mfcc_source()} Created...")
        


    def generate_spectrogram(self):
        pass


    def generate_wavelet(self):
        pass


    def get_mfcc_source(self):
        return f"{self._data_source_path}mfcc_{self._num_mfcc}_{self._num_fft}_{self._hop_length}.json"


    def run(self):
        X, y = self.load_mfcc()
        print(X.shape)
        print(y.shape)
    
    

model1 = Music_Model(512, 2048, 20)
model1.run()

# model2 = Music_Model(256, 4096, 20)
# model2.run()
