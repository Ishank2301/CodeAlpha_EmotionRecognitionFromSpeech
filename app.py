import os
import sys
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
import io
import tempfile

matplotlib.use("Agg")

from src.features.feature_extractor import load_audio_bytes, preprocess_for_model
from src.models.model_loader import load_artifacts, predict_emotion

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

EMOTION_META = {
    "neutral": {
        "emoji": "😐",
        "color": "#6B7280",
        "description": "Calm, expressionless speech",
    },
    "calm": {
        "emoji": "😌",
        "color": "#10B981",
        "description": "Relaxed, composed tone",
    },
    "happy": {
        "emoji": "😄",
        "color": "#F59E0B",
        "description": "Joyful, upbeat energy",
    },
    "sad": {
        "emoji": "😢",
        "color": "#3B82F6",
        "description": "Low energy, downward inflection",
    },
    "angry": {
        "emoji": "😠",
        "color": "#EF4444",
        "description": "High energy, tense vocal cords",
    },
    "fearful": {
        "emoji": "😨",
        "color": "#8B5CF6",
        "description": "Elevated pitch, trembling voice",
    },
    "disgust": {
        "emoji": "🤢",
        "color": "#84CC16",
        "description": "Guttural, aversive tone",
    },
    "surprised": {
        "emoji": "😲",
        "color": "#EC4899",
        "description": "Sudden pitch changes, wide range",
    },
}

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }

    .main { background: #0F1117; }
    .stApp { background: #0F1117; color: #E2E8F0; }

    .emotion-card {
        background: linear-gradient(135deg, #1E2433 0%, #252D40 100%);
        border: 1px solid #2D3748;
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .emotion-emoji  { font-size: 72px; margin-bottom: 8px; }
    .emotion-label  { font-family: 'Space Mono', monospace; font-size: 28px; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; }
    .confidence-bar { height: 8px; border-radius: 4px; margin: 16px 0 8px; }
    .confidence-text { font-size: 14px; color: #94A3B8; }

    .prob-row { display: flex; align-items: center; gap: 12px; margin: 6px 0; }
    .prob-label { font-size: 13px; width: 80px; color: #CBD5E0; font-family: 'Space Mono', monospace; }
    .prob-bar-bg { flex: 1; height: 6px; background: #2D3748; border-radius: 3px; overflow: hidden; }
    .prob-bar-fill { height: 100%; border-radius: 3px; transition: width 0.6s ease; }
    .prob-pct { font-size: 12px; color: #94A3B8; width: 40px; text-align: right; }

    .feature-badge {
        display: inline-block;
        background: #1E2433;
        border: 1px solid #374151;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        color: #94A3B8;
        margin: 3px;
        font-family: 'Space Mono', monospace;
    }
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 13px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #4B5563;
        margin-bottom: 12px;
    }
    div[data-testid="stFileUploader"] { background: #1E2433; border-radius: 12px; border: 1px dashed #374151; padding: 12px; }
    .stButton > button {
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        color: white; border: none; border-radius: 8px;
        padding: 10px 24px; font-family: 'Space Mono', monospace;
        font-size: 13px; letter-spacing: 1px; width: 100%;
    }
    .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

    .block-container {
        max-width: 1320px;
        padding-top: 32px;
        padding-bottom: 48px;
    }
    .visualization-wrap {
        margin-top: 32px;
        padding-top: 18px;
        border-top: 1px solid #1F2937;
    }
    div[data-testid="stVerticalBlock"] > div:has(> .section-header) {
        margin-bottom: 6px;
    }
    div[data-testid="stTabs"] button {
        color: #CBD5E0;
        font-family: 'Space Mono', monospace;
        font-size: 12px;
        letter-spacing: 1px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model artifacts...")
def get_artifacts():
    return load_artifacts()


with st.sidebar:
    st.markdown("## 🎙️ SER System")
    st.markdown("---")
    st.markdown("""
    **Model:** CNN + BiLSTM + Attention  
    **Dataset:** RAVDESS  
    **Features:** MFCC · Mel · Chroma · ZCR · RMS  
    **Classes:** 8 emotions  
    """)
    st.markdown("---")
    st.markdown("**Supported formats**")
    st.markdown("WAV · MP3 · OGG · FLAC · M4A")
    st.markdown("---")
    st.markdown(
        """
    <div style='color:#4B5563; font-size:12px'>
    Built with TensorFlow + Streamlit<br>
    Trained on Kaggle GPU (P100)
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("# 🎙️ Speech Emotion Recognition")
st.markdown(
    "<p style='color:#64748B; margin-top:-8px; margin-bottom:28px'>"
    "CNN + BiLSTM + Attention &nbsp;·&nbsp; RAVDESS Dataset &nbsp;·&nbsp; 8 Emotion Classes"
    "</p>",
    unsafe_allow_html=True,
)

col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown(
        "<div class='section-header'>Upload Audio</div>", unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Drop an audio file here",
        type=["wav", "mp3", "ogg", "flac", "m4a"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")
        st.markdown(
            f"<div style='font-size:13px; color:#6B7280; margin-top:8px'>"
            f"📄 {uploaded_file.name} &nbsp;·&nbsp; {uploaded_file.size / 1024:.1f} KB"
            f"</div>",
            unsafe_allow_html=True,
        )
        analyze_btn = st.button("⚡ Analyse Emotion")
    else:
        st.markdown(
            "<div style='color:#4B5563; font-size:14px; padding:16px 0'>"
            "Upload a .wav or .mp3 file to begin analysis."
            "</div>",
            unsafe_allow_html=True,
        )
        analyze_btn = False

    # Feature badges
    st.markdown(
        "<br><div class='section-header'>Extracted Features</div>",
        unsafe_allow_html=True,
    )
    for feat in [
        "MFCC (40 coeff)",
        "Δ-MFCC",
        "ΔΔ-MFCC",
        "Mel Spec (128)",
        "Chroma (12)",
        "ZCR",
        "RMS Energy",
    ]:
        st.markdown(
            f"<span class='feature-badge'>{feat}</span>", unsafe_allow_html=True
        )

with col_result:
    st.markdown("<div class='section-header'>Prediction</div>", unsafe_allow_html=True)

    if uploaded_file and analyze_btn:
        try:
            artifacts = get_artifacts()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

        with st.spinner("Extracting features & running inference..."):
            audio_bytes = uploaded_file.read()
            y = load_audio_bytes(audio_bytes)
            X = preprocess_for_model(y, artifacts["scaler"], artifacts["max_len"])
            result = predict_emotion(X, artifacts)

        emotion = result["emotion"]
        conf = result["confidence"]
        meta = EMOTION_META.get(
            emotion, {"emoji": "🎵", "color": "#6366F1", "description": ""}
        )

        # Emotion card
        st.markdown(
            f"""
        <div class='emotion-card'>
            <div class='emotion-emoji'>{meta['emoji']}</div>
            <div class='emotion-label' style='color:{meta["color"]}'>{emotion}</div>
            <div class='confidence-text' style='margin-top:6px'>{meta['description']}</div>
            <div class='confidence-bar' style='background: linear-gradient(90deg, {meta["color"]} {conf*100:.0f}%, #2D3748 {conf*100:.0f}%)'></div>
            <div class='confidence-text'>Confidence: <strong style='color:#E2E8F0'>{conf*100:.1f}%</strong></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Probability bars
        st.markdown(
            "<br><div class='section-header'>All Probabilities</div>",
            unsafe_allow_html=True,
        )
        sorted_probs = sorted(
            result["all_probs"].items(), key=lambda x: x[1], reverse=True
        )
        for emo, prob in sorted_probs:
            m = EMOTION_META.get(emo, {"color": "#6366F1"})
            st.markdown(
                f"""
            <div class='prob-row'>
                <div class='prob-label'>{EMOTION_META.get(emo, {}).get('emoji','🎵')} {emo}</div>
                <div class='prob-bar-bg'>
                    <div class='prob-bar-fill' style='width:{prob*100:.1f}%; background:{m["color"]}'></div>
                </div>
                <div class='prob-pct'>{prob*100:.1f}%</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    else:
        st.markdown(
            "<div style='color:#374151; font-size:14px; padding:48px 0; text-align:center'>"
            "🎤 &nbsp; Prediction will appear here after analysis"
            "</div>",
            unsafe_allow_html=True,
        )

if uploaded_file and analyze_btn:
    st.markdown(
        "<div class='visualization-wrap'><div class='section-header'>Audio Visualizations</div></div>",
        unsafe_allow_html=True,
    )
    viz_col1, viz_col2 = st.columns([1, 1], gap="large")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(audio_bytes)

    try:
        y_viz, _ = librosa.load(tmp_path, sr=22050, duration=3)

        with viz_col1:
            fig, ax = plt.subplots(figsize=(10, 4), facecolor="#1E2433")
            ax.set_facecolor("#1E2433")
            times = np.linspace(0, len(y_viz) / 22050, len(y_viz))
            ax.plot(times, y_viz, color="#6366F1", linewidth=1.2, alpha=0.95)
            ax.fill_between(times, y_viz, alpha=0.22, color="#6366F1")
            ax.set_title("Waveform", color="#CBD5E0", fontsize=13, pad=10)
            ax.tick_params(colors="#64748B", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#2D3748")
            ax.set_xlabel("Time (s)", color="#64748B", fontsize=10)
            ax.set_ylabel("Amplitude", color="#64748B", fontsize=10)
            fig.tight_layout(pad=1.6)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with viz_col2:
            fig, ax = plt.subplots(figsize=(10, 4), facecolor="#1E2433")
            ax.set_facecolor("#1E2433")
            mel = librosa.feature.melspectrogram(y=y_viz, sr=22050, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            librosa.display.specshow(
                mel_db,
                sr=22050,
                hop_length=512,
                x_axis="time",
                y_axis="mel",
                ax=ax,
                cmap="magma",
            )
            ax.set_title("Mel Spectrogram", color="#CBD5E0", fontsize=13, pad=10)
            ax.tick_params(colors="#64748B", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#2D3748")
            ax.set_xlabel("Time (s)", color="#64748B", fontsize=10)
            ax.set_ylabel("Frequency", color="#64748B", fontsize=10)
            fig.tight_layout(pad=1.6)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown(
            "<br><div class='section-header'>Feature Visualizations</div>",
            unsafe_allow_html=True,
        )
        mfcc_tab, mel_tab, chroma_tab, energy_tab = st.tabs(
            ["MFCC", "Mel", "Chroma", "ZCR + RMS"]
        )

        with mfcc_tab:
            mfcc = librosa.feature.mfcc(y=y_viz, sr=22050, n_mfcc=40)
            fig, ax = plt.subplots(figsize=(14, 5), facecolor="#1E2433")
            ax.set_facecolor("#1E2433")
            img = librosa.display.specshow(
                mfcc, sr=22050, x_axis="time", y_axis="mel", ax=ax, cmap="viridis"
            )
            ax.set_title("MFCC (40 coefficients)", color="#CBD5E0", fontsize=14, pad=12)
            ax.tick_params(colors="#64748B", labelsize=10)
            for spine in ax.spines.values():
                spine.set_color("#2D3748")
            cbar = plt.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.015)
            cbar.ax.tick_params(colors="#64748B")
            fig.tight_layout(pad=1.8)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with mel_tab:
            mel = librosa.feature.melspectrogram(y=y_viz, sr=22050, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            fig, ax = plt.subplots(figsize=(14, 5), facecolor="#1E2433")
            ax.set_facecolor("#1E2433")
            img = librosa.display.specshow(
                mel_db, sr=22050, x_axis="time", y_axis="mel", ax=ax, cmap="magma"
            )
            ax.set_title("Mel Spectrogram (128 bins)", color="#CBD5E0", fontsize=14, pad=12)
            ax.tick_params(colors="#64748B", labelsize=10)
            for spine in ax.spines.values():
                spine.set_color("#2D3748")
            cbar = plt.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.015)
            cbar.ax.tick_params(colors="#64748B")
            fig.tight_layout(pad=1.8)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with chroma_tab:
            chroma = librosa.feature.chroma_stft(y=y_viz, sr=22050)
            fig, ax = plt.subplots(figsize=(14, 4.5), facecolor="#1E2433")
            ax.set_facecolor("#1E2433")
            img = librosa.display.specshow(
                chroma, sr=22050, x_axis="time", y_axis="chroma", ax=ax, cmap="coolwarm"
            )
            ax.set_title(
                "Chroma STFT (12 pitch classes)", color="#CBD5E0", fontsize=14, pad=12
            )
            ax.tick_params(colors="#64748B", labelsize=10)
            for spine in ax.spines.values():
                spine.set_color("#2D3748")
            cbar = plt.colorbar(img, ax=ax, pad=0.015)
            cbar.ax.tick_params(colors="#64748B")
            fig.tight_layout(pad=1.8)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with energy_tab:
            zcr = librosa.feature.zero_crossing_rate(y_viz)[0]
            rms = librosa.feature.rms(y=y_viz)[0]
            frames = range(len(zcr))
            t = librosa.frames_to_time(frames, sr=22050)

            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(14, 6.5), facecolor="#1E2433", sharex=True
            )

            ax1.set_facecolor("#1E2433")
            ax1.fill_between(t, zcr, alpha=0.45, color="#10B981", label="ZCR")
            ax1.plot(t, zcr, color="#34D399", linewidth=1.8)
            ax1.set_title("Zero Crossing Rate (ZCR)", color="#CBD5E0", fontsize=13, pad=10)
            ax1.set_ylabel("ZCR", color="#64748B")
            ax1.tick_params(colors="#64748B", labelsize=10)
            for spine in ax1.spines.values():
                spine.set_color("#2D3748")
            ax1.legend(
                loc="upper right",
                facecolor="#1E2433",
                edgecolor="#2D3748",
                labelcolor="#CBD5E0",
            )

            ax2.set_facecolor("#1E2433")
            ax2.fill_between(t, rms, alpha=0.45, color="#3B82F6", label="RMS Energy")
            ax2.plot(t, rms, color="#60A5FA", linewidth=1.8)
            ax2.set_title("RMS Energy", color="#CBD5E0", fontsize=13, pad=10)
            ax2.set_xlabel("Time (s)", color="#64748B", fontsize=10)
            ax2.set_ylabel("RMS", color="#64748B")
            ax2.tick_params(colors="#64748B", labelsize=10)
            for spine in ax2.spines.values():
                spine.set_color("#2D3748")
            ax2.legend(
                loc="upper right",
                facecolor="#1E2433",
                edgecolor="#2D3748",
                labelcolor="#CBD5E0",
            )

            fig.tight_layout(pad=1.8)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
