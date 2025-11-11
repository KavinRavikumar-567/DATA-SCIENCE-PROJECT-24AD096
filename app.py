import streamlit as st
import numpy as np
import joblib
import librosa
import sounddevice as sd # Used for capturing microphone input
import tempfile
import soundfile as sf # Used to save recorded audio to a file
from tensorflow.keras.models import load_model

# --- Configuration ---
RECORDING_DURATION = 3 # seconds
SAMPLE_RATE = 22050
N_MFCC = 40

# --- 1. Load Model, Scaler, and Encoder (Cached for Speed) ---

@st.cache_resource
def load_all_components():
    try:
        model = load_model("emotion_model.keras") 
        le = joblib.load('label_encoder.joblib')
        scaler = joblib.load('scaler.joblib') 
        return model, le, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.stop()

# --- 2. Feature Extraction Function (Must Match Training) ---

def extract_features(audio_path):
    """Loads audio from path and extracts MFCC features."""
    try:
        y_audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=RECORDING_DURATION)
        
        # Trim silence (optional)
        y_audio, _ = librosa.effects.trim(y_audio)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        # Reshape for the scaler/model (1 sample, 40 features)
        return mfcc_mean.reshape(1, -1)
    except Exception as e:
        st.error(f"Error during feature extraction: {e}")
        return None

# --- 3. Prediction Function ---

def predict_emotion(features, model, scaler, le):
    """Scales features and returns predicted emotion and confidence."""
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Predict probabilities
    probs = model.predict(features_scaled, verbose=0)[0]
    
    # Get the index of the highest probability
    pred_index = np.argmax(probs)
    
    # Decode to emotion name
    predicted_emotion = le.inverse_transform([pred_index])[0]
    confidence = probs[pred_index] * 100
    
    return predicted_emotion, confidence

# --- 4. Streamlit UI Logic ---

def main():
    st.set_page_config(page_title="Real-Time Voice Emotion Detector", layout="centered")
    st.title("🎙️ Live Voice Emotion Recognition")
    st.markdown("Press **Record** and speak for **3 seconds** to analyze your current emotion.")

    # Load all saved components
    model, le, scaler = load_all_components()
    
    # Emotion-to-Emoji Mapping
    emoji_map = {"happy": "😃", "sad": "😢", "angry": "😡", "neutral": "😐", 
                 "fearful": "😨", "disgust": "🤢", "surprised": "😲", "calm": "😌"}

    # Main Record Button
    if st.button("🔴 Start Recording (3 Seconds)", key="record_button", type="primary"):
        
        # 4a. Record Audio
        with st.spinner(f"Recording in progress... Speak now for {RECORDING_DURATION} seconds!"):
            # Record audio using sounddevice
            recording = sd.rec(int(RECORDING_DURATION * SAMPLE_RATE), 
                               samplerate=SAMPLE_RATE, 
                               channels=1, 
                               dtype='float32')
            sd.wait() 
            
            # Save recording to a temporary WAV file for librosa to load
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, recording.flatten(), SAMPLE_RATE)
                temp_audio_path = tmp_file.name

        st.success("Recording complete! Analyzing...")

        # 4b. Display Audio and Analyze
        st.audio(temp_audio_path, format='audio/wav')
        
        with st.spinner("Extracting features and predicting emotion..."):
            
            # --- Feature Extraction and Prediction ---
            features = extract_features(temp_audio_path)

            if features is not None and features.shape[1] == N_MFCC:
                emotion, confidence = predict_emotion(features, model, scaler, le)
                
                # --- Display Results ---
                emoji = emoji_map.get(emotion, "❓")
                
                st.subheader("Prediction Result:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label="Predicted Emotion", value=f"{emotion.upper()} {emoji}")
                
                with col2:
                    st.metric(label="Confidence", value=f"{confidence:.2f}%")
                
                # Visual feedback
                st.markdown("---")
                st.caption(f"Confidence Level for {emotion.upper()}:")
                st.progress(float(confidence / 100))
            else:
                 st.warning("Could not extract expected features. Audio might be too short or silent.")


if __name__ == '__main__':
    main()