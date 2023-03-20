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
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt


class Music_Model:
    
    def __init__(self, hop_length=512, num_fft=2048, num_mfcc=20):
        # Default Values
        # 2500 for small (S), 10133 for large (L)
        self._total = 2500
        self._tag = "S"
        self._moods = ["aggressive", "dramatic", "happy", "romantic", "sad"]
        self._audio_source_path = f"music_mood_{self._tag}/"
        self._data_source_path = "ProcessedData/"
        self._audio_class_source = f"{self._data_source_path}baseInfo_{self._tag}.csv"
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
        

    def get_mfcc_source(self):
        return f"{self._data_source_path}mfcc_{self._num_mfcc}_{self._num_fft}_{self._hop_length}_{self._tag}.json"


    def generate_spectrogram(self):
        pass


    def generate_wavelet(self):
        pass


    def plot_NN_history(self, history):
        fig, axs = plt.subplots(2)
        axs[0].plot(history.history["accuracy"], label="training")
        axs[0].plot(history.history["val_accuracy"], label="validation")
        axs[0].set_ylabel("Accuracy")
        axs[0].legend()

        axs[1].plot(history.history["loss"], label="training")
        axs[1].plot(history.history["val_loss"], label="validation")
        axs[1].set_ylabel("Error")
        axs[1].set_xlabel("Epoch")
        axs[1].legend()
        plt.show()


    def split_NN_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test


    def _NN(self, model, X, y, name):
        print(f"Training {name}...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_NN_data(X, y)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=1)
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"{name} Accuracy: {test_acc}")
        # self.plot_NN_history(history)


    def normal_NN(self, X, y):
        model = Sequential([
            Flatten(input_shape=(X.shape[1], X.shape[2])),
            Dense(512, activation='relu', kernel_regularizer=l2(0.02)),
            Dense(256, activation='relu', kernel_regularizer=l2(0.02)),
            Dense(128, activation='relu', kernel_regularizer=l2(0.02)),
            Dense(len(self._moods), activation="softmax")
        ])
        self._NN(model, X, y, "Normal NN")

    
    def lstm_NN(self, X, y):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2]), kernel_regularizer=l2(0.01)),
            LSTM(64),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(len(self._moods), activation="softmax")
        ])
        self._NN(model, X, y, "LSTM NN")


    def cnn(self, X, y):
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding="valid", input_shape=X.shape[1:]),
            MaxPooling2D(2, padding="same"),
            Conv2D(128, (3, 3), activation='relu', padding="valid"),
            MaxPooling2D(2, padding="same"),
            Dropout(0.3),
            Conv2D(128, (3, 3), activation='relu', padding="valid"),
            MaxPooling2D(2, padding="same"),
            Dropout(0.3),
            GlobalAveragePooling2D(),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(len(self._moods), activation="softmax")
        ])
        self._NN(model, X, y, "CNN")


    def run_mfcc_NN(self):
        X, y = self.load_mfcc()
        self.normal_NN(X, y)
        self.lstm_NN(X, y)
        self.cnn(X, y)


    def run(self):
        self.run_mfcc_NN()
        
    

model1 = Music_Model(512, 2048, 20)
model1.run()

# model2 = Music_Model(256, 4096, 20)
# model2.run()
