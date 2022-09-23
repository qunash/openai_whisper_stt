import os
import gradio as gr
import whisper
import time

model = whisper.load_model("base")

def transcribe(audio, state="", delay=0.2):
    time.sleep(delay)
    result = model.transcribe(audio, language="english")
    state += result['text'] + " "
    # return f"Language: {result['language']}\
    #         \n\nText: {state}"
    return state, state

def debug(audio, state="", delay=0.2):
  print(whisper.load_audio(audio).shape)
  state += str(whisper.load_audio(audio))
  # print(state)
  return state, state


delay_slider = gr.inputs.Slider(minimum=0, maximum=10, default=0.2, label="Delay (seconds). The rate of transcription (1 sec + delay).")

title = "OpenAI's Whisper Real-time Demo"

gr.Interface(
    fn=transcribe,
    # fn=debug,
    inputs=[
        # gr.Audio(source="upload", type="filepath"),
        gr.Audio(source="microphone", type="filepath", streaming=True),
        "state",
        delay_slider
        ], 
    outputs=[
        gr.Textbox(label="Transcription", lines=10, max_lines=20),
        "state"
    ],
    live=True,
    allow_flagging='never',
    title=title,
).launch(enable_queue=True)