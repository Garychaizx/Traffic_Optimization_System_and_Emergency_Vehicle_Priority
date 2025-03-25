import numpy as np
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
import os

# Custom LSTM layer to handle the time_major argument
class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)  # Remove the time_major argument
        super(CustomLSTM, self).__init__(*args, **kwargs)

# 1️⃣ Load the trained LSTM model
try:
    model = load_model('siren_model/best_model.keras')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
# model = load_model('models/best_model.keras')

# 2️⃣ Function to extract features from audio
def extract_features(audio_file, max_pad_len=862):
    try:
        # Debug: Check if file exists
        if not os.path.exists(audio_file):
            print(f"File not found: {audio_file}")
            return None

        # Load the audio file
        audio, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
        print(f"✅ Audio loaded successfully: {audio_file}")
        print(f"Sample rate: {sample_rate}")
        print(f"Audio shape: {audio.shape}")

        # Extract 80 MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
        print(f"MFCCs shape: {mfccs.shape}")

        # Pad or truncate to match the expected input size
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        print(f"MFCCs after padding/truncation: {mfccs.shape}")

        return mfccs
    except Exception as e:
        print(f"Error encountered while parsing file {audio_file}: {e}")
        return None

# 3️⃣ Path to the test audio file
test_audio_file = "dynamic_sounds/sound-effect-uk-ambulance-siren-164354.wav"  # Replace with your audio file path

# 4️⃣ Extract features
features = extract_features(test_audio_file)
if features is not None:
    # Reshape features to match the model's input shape
    features = np.mean(features, axis=1)  # Compress along time axis
    features = features.reshape(1, 1, 80)  # Reshape to (1, 1, 80)
    print(f"Reshaped features: {features.shape}")

    # 5️⃣ Make prediction
    prediction = model.predict(features)
    print(f"Raw prediction: {prediction}")

    # 6️⃣ Interpret results
    if prediction[0][0] > 0.5:  # Assuming binary classification (e.g., siren vs. no siren)
        print("🚨 Siren Detected!")
    else:
        print("✅ No Siren Detected.")
else:
    print("⚠️ Could not extract features from the audio.")