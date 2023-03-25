import gradio as gr


def predict_audio(audio):
    return true

demo = gr.Interface(fn=predict_audio, inputs=gr.Audio(type="filepath"), outputs="text")

demo.launch()