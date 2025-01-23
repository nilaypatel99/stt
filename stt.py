import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
import io
import tempfile
import torch
from model import model
from pathlib import Path


# Check for GPU availability
device_option = "GPU" if torch.cuda.is_available() else "CPU"
device = "cuda" if device_option == "GPU" else "cpu"

# Page title
st.title("🎙️ Speech-to-Text Transcription")
# Sidebar
st.sidebar.title("Settings")
user_options = st.sidebar.selectbox(
    "Menu", ["Select One", "Upload_Audio", "Record_Audio"]
)
st.sidebar.info(f"Detected Device: {device_option}")

# Load the ASR model only once
asr_model = model.load_asr_model(device)

audio_bytes = None

# --- Option 1: Upload audio ---
if user_options == "Upload_Audio":
    uploaded_file = st.file_uploader(
        "📂 Upload an audio file (WAV, MP3)", type=["wav", "mp3"]
    )
    if uploaded_file:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format="audio/wav")

# --- Option 2: Record audio ---
elif user_options == "Record_Audio":
    audio_data = audio_recorder("Click here to Record")
    if audio_data:
        audio_bytes = audio_data
        st.audio(audio_bytes, format="audio/wav")

# --- Perform transcription if we have valid audio bytes ---
if audio_bytes:
    with st.spinner("🔄 Transcribing audio... Please wait."):
        try:
            # Preprocess the audio (returns BytesIO)
            processed_audio = model.preprocess_audio(io.BytesIO(audio_bytes))
            # using temporary storage
            fd, raw_path = tempfile.mkstemp(
                suffix=".wav", dir="C:/Users/HP/AppData/Local/Temp"
            )
            # Close the file descriptor
            os.close(fd)

            # Write processed audio to the temp file
            with open(raw_path, "wb") as f:
                f.write(processed_audio.getvalue())

            # Converting to forward slashes just in case
            final_path = str(Path(raw_path).resolve()).replace("\\", "/")

            # calling SpeechBrain to transcribe
            try:
                transcript = model.convert_speech_to_text(final_path, asr_model)
            finally:
                # Clean up the file after transcription
                if os.path.exists(raw_path):
                    os.remove(raw_path)

            st.success("✅ Transcription completed successfully!")
            # Display transcription
            st.text_area("Transcribed Text:", transcript, height=200)

            # Download button
            st.download_button(
                label="📥 Download Transcription",
                data=transcript,
                file_name="transcription.txt",
                mime="text/plain",
            )
        except Exception as e:
            st.error(f"❌ Error processing audio: {e}")
else:
    st.markdown("🎙️ No audio recorded or uploaded yet.")

st.markdown("---")
