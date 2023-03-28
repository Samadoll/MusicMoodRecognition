"""
proposal: Have you chosen a correct background music?
"""
import os
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import json
import glob
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import tensorflow as tf
from sklearn import preprocessing
import warnings
import time


class Music_Model:
    
    def __init__(self, hop_length=512, num_fft=2048, num_mfcc=20):
        # Default Values
        self._moods = ["happy", "sad"] # ["aggressive", "dramatic", "happy", "romantic", "sad"]
        # 500 for small (S), >500 for large (L)
        self._tag = "L"
        self._files_per_feat = 1000
        self._total = self._files_per_feat * len(self._moods)
        self._output_layer_activation = "sigmoid" # "softmxax"
        self._NN_loss_func = "binary_crossentropy" # "sparse_categorical_crossentropy"
        self._output_layer_dim = 1
        self._audio_source_path = f"music_mood_{self._tag}/"
        self._data_source_path = "ProcessedData/"
        self._mel_spectrograms_path = "ProcessedData/melspectrograms/"
        self._mfcc_visual_path = "ProcessedData/mfccVisual/"
        self._audio_class_source = f"{self._data_source_path}baseInfo_{self._tag}.csv"
        self._sample_rate = 44100
        self._plot_path = f"ProcessedData/plots/{str(time.time())}/"

        # Variables
        self._audio_class_info = self.load_class_info()
        self._hop_length = hop_length
        self._num_fft = num_fft
        self._num_mfcc = num_mfcc
        self._current_feat = ""

    #####################################
    #           Data Process            #
    ##################################### 

    def load_class_info(self):
        if not os.path.exists(self._audio_class_source):
            self.create_class_info()
        return pd.read_csv(self._audio_class_source)


    def create_class_info(self):
        with open(self._audio_class_source, "w") as info_file:
            info_file.write("audio_name,audio_path,mood\n")
            for c in self._moods:
                audio_path = f"{self._audio_source_path}{c}/"
                audios = os.listdir(audio_path) if self._files_per_feat == 500 else random.sample(os.listdir(audio_path), self._files_per_feat)
                for audio in audios:
                    info_file.write(f"{audio},{audio_path}/{audio},{c}\n")
        print(f"{self._audio_class_source} Created...")


    #####################################
    #     Data Process - Feature        #
    ##################################### 

    def load_feature(self, path, feat_name):
        data = None
        if not os.path.exists(path):
            data = self.extract_feature(path, feat_name)
        if data == None:
            with open(path, "r") as feat_file:
                data = json.load(feat_file)
        feat = np.array(data[feat_name])
        moods = np.array(list(map(self._moods.index, data["mood"])))
        print(f"{path} Loaded...")
        return feat, moods


    def extract_feature(self, feat_path, feat_name):
        data = {
            feat_name: [],
            "mood": []
        }
        with tqdm(total=self._total, desc=f"Processing {feat_name}") as pbar:
            for audio, path, label in self._audio_class_info.itertuples(index=False):
                signal, sample_rate = librosa.load(path, sr=self._sample_rate)
                feat = self.get_feat(feat_name, signal, sample_rate)
                data[feat_name].append(feat)
                data["mood"].append(label)
                pbar.update(1)
        
        with open(feat_path, "w") as feat_file:
            json.dump(data, feat_file, indent=2)
        print(f"{feat_path} Created...")
        return data


    def get_feat(self, feat_name, signal, sample_rate):
        if feat_name == "mfcc":
            return librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=self._num_fft, n_mfcc=self._num_mfcc, hop_length=self._hop_length).T.tolist()
        if feat_name == "mel_spec":
            return librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=self._num_fft, hop_length=self._hop_length).T.tolist()
        if feat_name == "mel_mfcc":
            mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=self._num_fft, n_mfcc=self._num_mfcc, hop_length=self._hop_length)
            mel = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=self._num_fft, hop_length=self._hop_length)
            combined = np.concatenate((mfcc, mel), axis=0)
            return combined.T.tolist()
        if feat_name == "multifeat":
            return self.get_multifeat(signal, sample_rate).tolist()
        if feat_name == "feat_mean_var":
            return self.get_feat_mean_var(signal, sample_rate).tolist()


    def get_multifeat(self, signal, sample_rate):
        zero_cross = librosa.feature.zero_crossing_rate(y=signal, hop_length=self._hop_length)
        mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=self._num_fft, n_mfcc=self._num_mfcc, hop_length=self._hop_length)
        mel = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=self._num_fft, hop_length=self._hop_length)
        stft = np.abs(librosa.stft(y=signal))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate, n_fft=self._num_fft, hop_length=self._hop_length)
        combined = np.concatenate((zero_cross, mfcc, mel, chroma), axis=0)
        return combined.T


    def get_feat_mean_var(self, signal, sample_rate):
        zero_cross = librosa.feature.zero_crossing_rate(y=signal, hop_length=self._hop_length).T
        zero_cross_mean = np.mean(zero_cross, axis=0)
        zero_cross_var = np.var(zero_cross, axis=0)
        mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=self._num_fft, n_mfcc=self._num_mfcc, hop_length=self._hop_length).T
        mfcc_mean = np.mean(mfcc, axis=0)
        mfcc_var = np.var(mfcc, axis=0)
        mel = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=self._num_fft, hop_length=self._hop_length).T
        mel_mean = np.mean(mel, axis=0)
        mel_var = np.var(mel, axis=0)
        stft = np.abs(librosa.stft(y=signal))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate, n_fft=self._num_fft, hop_length=self._hop_length).T
        chroma_mean = np.mean(chroma, axis=0)
        chroma_var = np.var(chroma, axis=0)
        return np.concatenate((zero_cross_mean, zero_cross_var, mfcc_mean, mfcc_var, mel_mean, mel_var, chroma_mean, chroma_var), axis=0)
            

    def get_json_source(self, feat_name):
        if feat_name == "mel_spec":
            return f"{self._data_source_path}mel_spec_{self._num_fft}_{self._hop_length}_{self._tag}.json"
        return f"{self._data_source_path}{feat_name}_{self._num_mfcc}_{self._num_fft}_{self._hop_length}_{self._tag}.json"


    #####################################
    #     Data Process - Images         #
    ##################################### 

    def generate_visual(self, feat_name):
        img_folder = self.get_feat_visual_path(feat_name)
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        count = 0
        with tqdm(total=self._total, desc=f"Generating {feat_name} Visual...") as pbar:
            for audio, path, label in self._audio_class_info.itertuples(index=False):
                count += 1
                self.generate_img_helper(img_folder, path, label, count, feat_name)
                pbar.update(1)


    def get_feat_visual_path(self, feat_name):
        if feat_name == "mel":
            return f"{self._mel_spectrograms_path}all/"
        if feat_name == "mfcc":
            return f"{self._mfcc_visual_path}scaled/"

    
    def get_feat_visual_split_path(self, feat_name):
        if feat_name == "mel":
            return self._mel_spectrograms_path
        if feat_name == "mfcc":
            return self._mfcc_visual_path


    def split_images(self, feat_name):
        X_train, X_test, y_train, y_test = train_test_split(self._audio_class_info.iloc[:,:-1], self._audio_class_info.iloc[:,-1], test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        self.split_images_helper(X_train["audio_path"], y_train, "train", feat_name)
        self.split_images_helper(X_val["audio_path"], y_val, "valid", feat_name)
        self.split_images_helper(X_test["audio_path"], y_test, "test", feat_name)


    def split_images_helper(self, x, y, tag, feat_name):
        count = 0
        with tqdm(total=len(x), desc=f"Generating {tag} {feat_name} Visual...") as pbar:
            for path, label in zip(x, y):
                img_folder = f"{self.get_feat_visual_split_path(feat_name)}{tag}/{label}/"
                if not os.path.exists(img_folder):
                    os.makedirs(img_folder)
                count += 1
                self.generate_img_helper(img_folder, path, label, count, feat_name)
                pbar.update(1)


    def generate_img_helper(self, folder, path, label, count, feat_name):
        signal, sample_rate = librosa.load(path, sr=self._sample_rate)
        
        if feat_name == "mel":
            mel = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=self._num_fft, hop_length=self._hop_length)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            librosa.display.specshow(mel_db, sr=sample_rate, fmax=8000)
        elif feat_name == "mfcc":
            mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=self._num_fft, n_mfcc=self._num_mfcc, hop_length=self._hop_length)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mfcc_scale = preprocessing.scale(mfcc, axis=1)
            librosa.display.specshow(mfcc_scale, sr=sample_rate)

        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        fig = plt.gcf()
        fig.savefig(f'{folder}{label}_{count}.png')
        plt.clf()
        plt.close()


    #####################################
    #      Model - Feature-based        #
    ##################################### 

    def plot_NN_history(self, history, name, title):
        if not os.path.exists(self._plot_path):
            os.makedirs(self._plot_path)
        fig, ax = plt.subplots()
        ax.plot(history.history["accuracy"], label="training")
        ax.plot(history.history["val_accuracy"], label="validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        plt.savefig(f"{self._plot_path}{name}_accuracy.png")

        fig, ax = plt.subplots()
        ax.plot(history.history["loss"], label="training")
        ax.plot(history.history["val_loss"], label="validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.savefig(f"{self._plot_path}{name}_loss.png")
        plt.clf()
        plt.close()


    def split_NN_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test


    def _NN(self, model, X, y, name):
        print(f"Training {name}...")
        callback = EarlyStopping(patience=10)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_NN_data(X, y)
        model.compile(loss=self._NN_loss_func, optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        # model.summary()
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=32, verbose=1)
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"{name} Accuracy: {test_acc}")
        self.plot_NN_history(history, name.replace(" ", "_"), f"{name}\nacc: {'{:.2f}'.format(test_acc * 100)}%, loss: {test_loss}")
        return test_loss, test_acc


    def normal_NN_1D(self, X, y):
        model = Sequential([
            Dense(512, activation='relu', input_shape=(X.shape[1],), kernel_regularizer=l2(0.02)),
            Dropout(0.4),
            Dense(256, activation='relu', kernel_regularizer=l2(0.02)),
            Dropout(0.4),
            Dense(128, activation='relu', kernel_regularizer=l2(0.02)),
            Dropout(0.4),
            Dense(self._output_layer_dim, activation=self._output_layer_activation)
        ])
        return self._NN(model, X, y, f"{self._current_feat} Normal NN")


    def normal_NN(self, X, y):
        model = Sequential([
            Flatten(input_shape=(X.shape[1], X.shape[2])),
            Dense(128, activation='relu', kernel_regularizer=l2(0.02)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu', kernel_regularizer=l2(0.02)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(512, activation='relu', kernel_regularizer=l2(0.02)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self._output_layer_dim, activation=self._output_layer_activation)
        ])
        return self._NN(model, X, y, f"{self._current_feat} Normal NN")

    
    def lstm_NN(self, X, y):
        # model 3
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2]), kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            LSTM(128, kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(self._output_layer_dim, activation=self._output_layer_activation)
        ])
        # model 4
        # model = Sequential([
        #     LSTM(256, return_sequences=True, input_shape=(X.shape[1], X.shape[2]), kernel_regularizer=l2(0.02)),
        #     Dropout(0.3),
        #     LSTM(128, return_sequences=True, kernel_regularizer=l2(0.02)),
        #     Dropout(0.3),
        #     LSTM(64, kernel_regularizer=l2(0.01)),
        #     Dropout(0.3),
        #     Dense(64, activation='relu', kernel_regularizer=l2(0.02)),
        #     Dropout(0.3),
        #     Dense(128, activation='relu', kernel_regularizer=l2(0.02)),
        #     Dropout(0.3),
        #     Dense(128, activation='relu', kernel_regularizer=l2(0.02)),
        #     Dropout(0.3),
        #     Dense(256, activation='relu', kernel_regularizer=l2(0.02)),
        #     Dropout(0.3),
        #     Dense(self._output_layer_dim, activation=self._output_layer_activation)
        # ])
        return self._NN(model, X, y, f"{self._current_feat} LSTM NN")


    def cnn(self, X, y):
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        model = Sequential([
            Conv2D(64, (2, 2), activation='relu', padding="valid", input_shape=X.shape[1:]),
            MaxPooling2D(2, padding="same"),
            Conv2D(128, (2, 2), activation='relu', padding="valid"),
            MaxPooling2D(2, padding="same"),
            Conv2D(256, (2, 2), activation='relu', padding="valid"),
            MaxPooling2D(2, padding="same"),
            Dropout(0.3),
            GlobalAveragePooling2D(),
            Dense(64, activation='relu'),
            Dense(128, activation='relu'),
            Dense(256, activation='relu'),
            Dense(self._output_layer_dim, activation=self._output_layer_activation)
        ])
        return self._NN(model, X, y, f"{self._current_feat} CNN 2D")


    def cnn_1d(self, X, y):
        drop = 0.3
        model = Sequential([
            Conv1D(64, 3, activation='relu', padding="valid", kernel_regularizer=l2(0.02), input_shape=X.shape[1:]),
            MaxPooling1D(3, padding="same"),
            Dropout(drop),
            Conv1D(128, 3, activation='relu', padding="valid", kernel_regularizer=l2(0.02)),
            MaxPooling1D(3, padding="same"),
            Dropout(drop),
            Conv1D(256, 3, activation='relu', padding="valid", kernel_regularizer=l2(0.02)),
            MaxPooling1D(3, padding="same"),
            Dropout(drop),
            Conv1D(512, 3, activation='relu', padding="valid", kernel_regularizer=l2(0.02)),
            MaxPooling1D(3, padding="same"),
            Dropout(drop),
            GlobalAveragePooling1D(),
            Dense(128, activation='relu', kernel_regularizer=l2(0.02)),
            Dropout(drop),
            Dense(256, activation='relu', kernel_regularizer=l2(0.02)),
            Dropout(drop),
            Dense(512, activation='relu', kernel_regularizer=l2(0.02)),
            Dropout(drop),
            Dense(self._output_layer_dim, activation=self._output_layer_activation)
        ])
        return self._NN(model, X, y, f"{self._current_feat} CNN 1D")


    def run_NN(self, X, y):
        return [self.normal_NN(X, y), self.lstm_NN(X, y), self.cnn(X, y), self.cnn_1d(X, y)] # [self.normal_NN(X, y), self.lstm_NN(X, y), self.cnn(X, y)]


    def run_feat_NN(self, feat_name):
        self._current_feat = feat_name
        if feat_name == "feat_mean_var":
            X, y = self.load_feature(self.get_json_source(feat_name), feat_name)
            return [self.normal_NN_1D(X, y)]
        else:
            X, y = self.load_feature(self.get_json_source(feat_name), feat_name)
            return self.run_NN(X, y)


    #####################################
    #        Model - Image-based        #
    #####################################
    #  ImageGenerator Not Good for This #
    #####################################
    def img_ImageGenerator_pre(self, feat_name):
        visual_path = self.get_feat_visual_split_path(feat_name)
        if not os.path.exists(visual_path):
            self.split_images(feat_name)

        batch_size = 16
        train_size = self._total * 0.8 * 0.8
        val_size = self._total * 0.8 * 0.2
        test_size = self._total * 0.2
        target_dim = (640, 480)
        
        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        train_set = train_datagen.flow_from_directory(f"{visual_path}train", target_size=target_dim, batch_size=batch_size, class_mode='categorical')
        val_set = val_datagen.flow_from_directory(f"{visual_path}valid", target_size=target_dim, batch_size=batch_size, class_mode='categorical')
        test_set = test_datagen.flow_from_directory(f"{visual_path}test", target_size=target_dim, batch_size=batch_size, class_mode='categorical')
        
        return batch_size, train_size, val_size, test_size, target_dim, train_set, val_set, test_set


    def run_img_ImageGenerator_NN(self, feat_name):
        batch_size, train_size, val_size, test_size, target_dim, train_set, val_set, test_set = self.img_ImageGenerator_pre(feat_name)
        callback = EarlyStopping(patience=10)
        model = model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding="valid", input_shape=target_dim + (3, )),
            MaxPooling2D(2, padding="same"),
            Conv2D(64, (3, 3), activation='relu', padding="valid"),
            MaxPooling2D(2, padding="same"),
            Dropout(0.3),
            Conv2D(128, (3, 3), activation='relu', padding="valid"),
            MaxPooling2D(2, padding="same"),
            Dropout(0.3),
            GlobalAveragePooling2D(),
            Flatten(),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(self._output_layer_dim, activation=self._output_layer_activation)
        ])
        model.compile(loss=self._NN_loss_func, optimizer=RMSprop(learning_rate=0.0001), metrics=['accuracy'])
        history = model.fit(train_set, steps_per_epoch=train_size // batch_size, epochs=30, validation_data=val_set, validation_steps=val_size // batch_size, callbacks=[callback], verbose=2)
        loss, acc = model.evaluate(test_set, steps=test_size // batch_size)
        print(f"{feat_name} ImageDataGenerator CNN Accuracy: {acc}")
        return loss, acc

    
    def run_vgg16_ImageGenerator_NN(self, feat_name):
        batch_size, train_size, val_size, test_size, target_dim, train_set, val_set, test_set = self.img_ImageGenerator_pre(feat_name)
        model = self.get_vgg16_cnn_model(target_dim + (3,))
        model.compile(loss=self._NN_loss_func, optimizer=RMSprop(learning_rate=0.0001), metrics=['accuracy'])
        history = model.fit(train_set, steps_per_epoch=train_size // batch_size, epochs=30, validation_data=val_set, validation_steps=val_size // batch_size, verbose=2)
        loss, acc = model.evaluate(test_set, steps=test_size // batch_size)
        print(f"{feat_name} VGG16 ImageDataGenerator CNN Accuracy: {acc}")
        return loss, acc
        
    
    #####################################
    #     Model - Image-based  CNN      #
    #####################################

    def run_img_CNN(self, feat_name, method_name):
        self._current_feat = feat_name
        img_folder = self.get_feat_visual_path(feat_name)
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
            self.generate_visual(feat_name)
        imgs = []
        mood = []
        for img in glob.glob(f"{img_folder}*.png"):
            img_name = img.replace(img_folder, "")
            image=np.array(tf.keras.preprocessing.image.load_img(img, color_mode='rgb', target_size=(300,300)))
            imgs.append(image)
            mood.append(self._moods.index(img_name[0:img_name.index("_")]))
        X = np.array(imgs)
        y = np.array(mood)
        print("Data Loaded...")
        model = self.get_vgg16_cnn_img_model(X.shape[1:]) if method_name == "VGG16" else self.get_cnn_img_model(X.shape[1:])
        return self._NN(model, X, y, f"{feat_name} {method_name} Image_CNN")
        

    def get_vgg16_cnn_img_model(self, dim):
        vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=dim)
        model = Sequential([
            vgg16_model,
            Conv2D(32, (3, 3), activation='relu', padding="valid"),
            Dropout(0.3),
            Conv2D(64, (3, 3), activation='relu', padding="valid"),
            Dropout(0.3),
            Conv2D(128, (3, 3), activation='relu', padding="valid"),
            Dropout(0.3),
            GlobalAveragePooling2D(),
            Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(self._output_layer_dim, activation=self._output_layer_activation)
        ])
        for layer in vgg16_model.layers:
            layer.trainable = False
        return model


    def get_cnn_img_model(self, dim):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding="valid", input_shape=dim),
            Dropout(0.3),
            Conv2D(64, (3, 3), activation='relu', padding="valid"),
            Dropout(0.3),
            Conv2D(128, (3, 3), activation='relu', padding="valid"),
            Dropout(0.3),
            GlobalAveragePooling2D(),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(self._output_layer_dim, activation=self._output_layer_activation)
        ])
        return model


    def run_img_NN_helper(self, feat_name):
        self.run_img_CNN(feat_name, "VGG16")
        self.run_img_CNN(feat_name, "VGG16")

    
    def run_img(self):
        feats = ["mel", "mfcc"]
        for feat in feats:
            self.run_img_NN_helper(feat)

    #####################################
    #               Run                 #
    ##################################### 

    def run(self):
        feats = ["mfcc", "mel_spec", "mel_mfcc", "multifeat", "feat_mean_var"]
        data = {k: self.run_feat_NN(k) for k in feats}
        for k, v in data.items():
            print(f"{k}: {','.join([f'[loss: {nn[0]}, acc: {nn[1]}]' for nn in v])}")
    

model1 = Music_Model()
# model1.run_img()
model1.run()
