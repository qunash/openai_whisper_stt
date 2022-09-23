import os
import gradio as gr
import whisper
import time

model = whisper.load_model("base")

def transcribe(audio, state={}, delay=0.2, lang=None, translate=False):
    time.sleep(delay)

    transcription = model.transcribe(
        audio,
        language = lang if lang != "auto" else None
    )
    state['transcription'] += transcription['text'] + " "

    if translate:
        x = whisper.load_audio(audio)
        x = whisper.pad_or_trim(x)
        mel = whisper.log_mel_spectrogram(x).to(model.device)

        options = whisper.DecodingOptions(task = "translation")
        translation = whisper.decode(model, mel, options)

        state['translation'] += translation.text + " "

    return state['transcription'], state['translation'], state, f"detected language: {transcription['language']}"


title = "OpenAI's Whisper Real-time Demo"
description = "A simple demo of OpenAI's [**Whisper**](https://github.com/openai/whisper) speech recognition model."

delay_slider = gr.inputs.Slider(minimum=0, maximum=5, default=0.2, label="Rate of transcription (1 sec + this value)")
lang_dropdown = gr.inputs.Dropdown(choices=["auto", "english", "afrikaans",
                                            "albanian", "amharic", "arabic",
                                            "armenian", "assamese", "azerbaijani",
                                            "bashkir", "basque", "belarusian",
                                            "bengali", "bosnian", "breton",
                                            "bulgarian", "catalan", "chinese",
                                            "croatian", "czech", "danish",
                                            "dutch", "estonian", "faroese",
                                            "finnish", "french", "galician",
                                            "georgian", "german", "greek",
                                            "gujarati", "haitian creole", "hausa",
                                            "hawaiian", "hebrew", "hindi",
                                            "hungarian", "icelandic", "indonesian",
                                            "italian", "japanese", "javanese",
                                            "kannada", "kazakh", "khmer",
                                            "korean", "kyrgyz", "lao",
                                            "latin", "latvian", "lingala",
                                            "lithuanian", "luxembourgish", "macedonian",
                                            "malagasy", "malay", "malayalam",
                                            "maltese", "maori", "marathi",
                                            "mongolian", "myanmar", "nepali",
                                            "norwegian", "nyanja", "nynorsk",
                                            "occitan", "oriya", "pashto",
                                            "persian", "polish", "portuguese",
                                            "punjabi", "romanian", "russian",
                                            "sanskrit", "sardinian", "serbian",
                                            "shona", "sindhi", "sinhala",
                                            "slovak", "slovenian", "somali",
                                            "spanish", "sundanese", "swahili",
                                            "swedish", "tagalog", "tajik",
                                            "tamil", "tatar", "telugu",
                                            "thai", "tigrinya", "tibetan",
                                            "turkish", "turkmen", "ukrainian",
                                            "urdu", "uzbek", "vietnamese",
                                            "welsh", "xhosa", "yiddish",
                                            "yoruba"],
                                   label="Language", default="auto", type="value")

translate_checkbox = gr.inputs.Checkbox(label="Translate to English", default=False)



transcription_tb = gr.Textbox(label="Transcription", lines=10, max_lines=20)
translation_tb = gr.Textbox(label="Translation", lines=10, max_lines=20)
detected_lang = gr.outputs.HTML(label="Detected Language")

state = gr.State({"transcription": "", "translation": ""})

gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(source="microphone", type="filepath", streaming=True),
        state,
        delay_slider,
        lang_dropdown,
        translate_checkbox
        ], 
    outputs=[
        transcription_tb,
        translation_tb,
        state,
        detected_lang
    ],
    live=True,
    allow_flagging='never',
    title=title,
    description=description,
).launch(
    # enable_queue=True,
    # debug=True
  )