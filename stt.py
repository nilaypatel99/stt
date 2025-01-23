import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
import torch
from model import model

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

# Initialize audio_filename to avoid reference errors
audio_filename = None

# Load the ASR model only once
asr_model = model.load_asr_model(device)

# --- Option 1: Upload audio ---
if user_options == "Upload_Audio":
    uploaded_file = st.file_uploader(
        "📂 Upload an audio file (WAV, MP3)", type=["wav", "mp3"]
    )
    if uploaded_file:
        audio_filename = "uploaded_audio.wav"
        with open(audio_filename, "wb") as f:
            f.write(uploaded_file.read())
        st.audio(audio_filename, format="audio/wav")

# --- Option 2: Record audio ---
elif user_options == "Record_Audio":
    audio_data = audio_recorder("Click here to Record")
    if audio_data:
        audio_filename = "recorded_audio.wav"
        with open(audio_filename, "wb") as f:
            f.write(audio_data)
        st.audio(audio_filename, format="audio/wav")

# --- Perform transcription if we have a valid file ---
if audio_filename and os.path.exists(audio_filename):
    with st.spinner("🔄 Transcribing audio... Please wait."):
        try:
            # Preprocess the audio
            processed_audio = model.preprocess_audio(audio_filename)
            # Convert speech to text
            transcript = model.convert_speech_to_text(processed_audio, asr_model)
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
