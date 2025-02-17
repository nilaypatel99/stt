from speechbrain.inference.ASR import EncoderDecoderASR
from pydub import AudioSegment
import streamlit as st
import io, os
import math
import tempfile
from pathlib import Path


class model:
    @st.cache_resource
    def load_asr_model(device):
        return EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-conformer-transformerlm-librispeech",
            savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
            run_opts={"device": device},
        )

    def convert_speech_to_text(audio_path, asr_model):
        text = asr_model.transcribe_file(audio_path)
        return text

    def preprocess_audio(input_audio_path):
        audio = AudioSegment.from_file(input_audio_path, format="wav")
        audio = audio.set_frame_rate(16000).set_channels(1)
        # processed_audio_path = f"processed_{input_audio_path}"
        processed_audio = io.BytesIO()
        audio.export(processed_audio, format="wav")
        return processed_audio

    def convert_speech_to_text_from_buffer(audio_buffer, asr_model):
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with open(temp_path, "wb") as f:
            f.write(audio_buffer.getvalue())

        # Transcribe using the path
        text = asr_model.transcribe_file(temp_path)

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return text
