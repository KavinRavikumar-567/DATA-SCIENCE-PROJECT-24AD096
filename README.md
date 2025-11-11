🎙️ Voice Emotion Detection using Python

A machine learning–based system that identifies human emotions from voice recordings using audio processing and classification algorithms.

📌 Overview

The Voice Emotion Detection System is designed to recognize emotions such as Happy, Sad, Angry, Neutral, Fear, and more using speech input.
It uses Librosa for feature extraction and Machine Learning / Deep Learning models for classification.
The project also includes a Streamlit interface for easy interaction.

✅ Key Features

🎤 Record or Upload Voice to detect emotions

🔎 Extracts MFCC, Chroma & Mel-Spectrogram features

🤖 ML model (Random Forest / Neural Network) for emotion classification

⚡ Real-time prediction support

📊 Simple Streamlit UI for users

🧩 Open-source and fully customizable

🛠️ Tech Stack
Component	Technology
Programming Language	Python
Audio Processing	Librosa, Soundfile
ML Model	Scikit-learn / TensorFlow
UI Framework	Streamlit
Data Handling	NumPy, Pandas
📂 Project Structure
Voice-Emotion-Detection/
│── app.py                 # Streamlit UI
│── train_model.py         # Model training script
│── emotion_model.pkl      # Saved trained model
│── utils.py               # Feature extraction helpers
│── dataset/               # Audio dataset (RAVDESS/TESS)
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation

📥 Installation
✅ 1. Clone the Repository
git clone https://github.com/yourusername/voice-emotion-detection.git
cd voice-emotion-detection

✅ 2. Install Dependencies
pip install -r requirements.txt


Example requirements.txt:

streamlit
librosa
numpy
pandas
scikit-learn
soundfile
tensorflow

▶️ How to Run the Project
✅ Run Streamlit App
streamlit run app.py


The app will open automatically at:

👉 http://localhost:8501/

🎯 How It Works
✅ Step 1 — Voice Input

User uploads or records an audio file (WAV recommended).

✅ Step 2 — Audio Preprocessing

Noise removal, normalization, resampling.

✅ Step 3 — Feature Extraction

MFCC

Chroma

Mel-Spectrogram

✅ Step 4 — Classification

Model predicts an emotion label using extracted features.

✅ Step 5 — Output

Emotion result displayed along with confidence score.

📊 Model Training

To retrain the model:

python train_model.py


You can modify:

Number of features

Algorithm selection

Training dataset



Deep Learning (CNN / LSTM) for improved accuracy

Multilingual emotion recognition

Real-time integration with voice assistants

Mobile app version

Dashboard for continuous emotion tracking

🤝 Contributing

Contributions are welcome!
Feel free to submit Pull Requests or open Issues.
