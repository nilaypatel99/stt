import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
import io
import tempfile
import torch
import time
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

# Load the ASR model
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

# --- Transcription ---
if audio_bytes:
    with st.spinner("🔄 Transcribing audio... Please wait."):
        try:
            # Preprocess the audio (returns BytesIO at 16 kHz, mono)
            processed_audio = model.preprocess_audio(io.BytesIO(audio_bytes))

            # Write the preprocessed audio to a temporary file
            fd, raw_path = tempfile.mkstemp(
                suffix=".wav", dir="C:/Users/HP/AppData/Local/Temp"
            )
            os.close(fd)

            with open(raw_path, "wb") as f:
                f.write(processed_audio.getvalue())

            final_path = str(Path(raw_path).resolve()).replace("\\", "/")

            try:
                # ------------------------
                # SINGLE-SHOT TRANSCRIPTION
                # ------------------------
                start_time_og = time.time()
                transcript_og = model.convert_speech_to_text_from_buffer(
                    io.BytesIO(audio_bytes), asr_model
                )
                time_og = time.time() - start_time_og
                # ------------------------
                # CHUNK_BASED-SHOT TRANSCRIPTION
                # ------------------------
                start_time_single = time.time()
                transcript_single = model.convert_speech_to_text(final_path, asr_model)
                time_single = time.time() - start_time_single

            finally:

                if os.path.exists(raw_path):
                    os.remove(raw_path)

            # ------------------------
            # Display results
            # ------------------------
            st.success("✅ Transcription completed successfully!")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Chunk-Shot Transcription")
                st.write(f"**Time Taken:** {time_single:.2f} seconds")
                st.text_area("Transcript (Chunk-Shot):", transcript_single, height=200)
            with col2:
                st.subheader("Single-Based Transcription")
                st.write(f"**Time Taken:** {time_og:.2f} seconds")
                st.text_area("Transcript (Single-Shot):", transcript_og, height=200)

            # --- Download buttons ---
            st.download_button(
                label="📥 Download Single-Shot Transcription",
                data=transcript_single,
                file_name="single_shot_transcription.txt",
                mime="text/plain",
            )
            st.download_button(
                label="📥 Download Chunked Transcription",
                data=transcript_og,
                file_name="chunked_transcription.txt",
                mime="text/plain",
            )

        except Exception as e:
            st.error(f"❌ Error processing audio: {e}")
else:
    st.markdown("🎙️ No audio recorded or uploaded yet.")

st.markdown("---")
