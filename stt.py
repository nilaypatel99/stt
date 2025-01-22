import streamlit as st
from audio_recorder_streamlit import audio_recorder
from speechbrain.inference.ASR import EncoderDecoderASR
import os

# Custom styling
st.markdown(
    """
    <style>
        .main-title {
            font-size:40px !important;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }
        .subtitle {
            font-size:20px !important;
            font-weight: bold;
            color: #555;
            text-align: center;
        }
        .info-text {
            font-size:16px !important;
            color: #333;
            text-align: center;
        }
        .stButton>button {
            border: none;
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Page title
st.markdown(
    '<p class="main-title">🎙️ Speech-to-Text Transcription</p>', unsafe_allow_html=True
)
st.markdown(
    '<p class="subtitle">Convert Your Voice to Text Instantly</p>',
    unsafe_allow_html=True,
)

# Sidebar for options
st.sidebar.title("Settings")
device_option = st.sidebar.radio("Choose processing device:", ["CPU", "GPU"])
st.sidebar.info("Ensure you have clear audio for better transcription accuracy.")


# Function to convert speech to text using GPU or CPU
def convertSpeechToText(audio_path, device):
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-conformer-transformerlm-librispeech",
        savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
        run_opts={"device": device.lower()},  # Use selected device
    )
    text = asr_model.transcribe_file(audio_path)
    return text


# Main section layout
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        '<p class="info-text">🎧 Click below to record your voice:</p>',
        unsafe_allow_html=True,
    )
    audio = audio_recorder("Record")

with col2:
    uploaded_file = st.file_uploader(
        "📂 Or upload an audio file (WAV, MP3)", type=["wav", "mp3"]
    )

# Process audio if recorded or uploaded
if audio is not None or uploaded_file is not None:
    audio_filename = "audio.wav"

    if audio is not None:
        with open(audio_filename, "wb") as f:
            f.write(audio)
        st.audio(audio_filename, format="audio/wav")
        st.success("✅ Audio recorded successfully.")

    if uploaded_file is not None:
        audio_filename = "uploaded_audio.wav"
        with open(audio_filename, "wb") as f:
            f.write(uploaded_file.read())
        st.audio(audio_filename, format="audio/wav")
        st.success(f"✅ Uploaded file: {uploaded_file.name}")

    # Ensure the file was saved correctly
    if os.path.exists(audio_filename):
        st.markdown(
            '<p class="subtitle">Processing Audio...</p>', unsafe_allow_html=True
        )
        with st.spinner("🔄 Transcribing audio... Please wait."):
            try:
                device = "cuda" if device_option == "GPU" else "cpu"
                transcript = convertSpeechToText(audio_filename, device)

                st.markdown(
                    '<p class="subtitle">🎯 Transcription:</p>', unsafe_allow_html=True
                )
                st.success("✅ Transcription completed successfully!")

                # Display transcription in a text area
                st.text_area("Transcribed Text:", transcript, height=200)

                # Download button for transcription
                st.download_button(
                    label="📥 Download Transcription",
                    data=transcript,
                    file_name="transcription.txt",
                    mime="text/plain",
                )

            except Exception as e:
                st.error(f"❌ Error processing audio: {e}")
    else:
        st.error("❌ Failed to save audio file.")

else:
    st.markdown(
        '<p class="info-text">🎙️ No audio recorded or uploaded yet.</p>',
        unsafe_allow_html=True,
    )

# Footer
st.markdown("---")
