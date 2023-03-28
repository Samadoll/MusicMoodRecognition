import gradio as gr
from tensorflow import keras
import numpy as np
import librosa

model = keras.models.load_model("model/model.h5")

def process_audio(audio):
    signal, sample_rate = librosa.load(audio, sr=44100)
    mel = np.array([librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=2048, hop_length=512).T.tolist()])
    return [mel, mel.reshape(mel.shape[0], mel.shape[1], mel.shape[2], 1)]

def predict_audio(audio):
    x = process_audio(audio)
    y = model.predict(x)[0][0]
    return "sad" if y > 0.5 else "happy"

demo = gr.Interface(fn=predict_audio, inputs=gr.Audio(type="filepath"), outputs="text")

demo.launch()