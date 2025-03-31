from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
from werkzeug.utils import secure_filename
import os
import librosa
import traceback  # Added for better error reporting

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
def load_models():
    global scaler, model, genre_mapping
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    model = pickle.load(open('models/model_svm.pkl', 'rb'))
    genre_mapping = pickle.load(open('models/genre_mapping.pkl', 'rb'))
    
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    
    # Initialize all 58 features
    features = {
        # Length (1)
        'length': float(len(y)/sr),
        
        # Chroma (2)
        'chroma_stft_mean': 0.0, 'chroma_stft_var': 0.0,
        
        # RMS (2)
        'rms_mean': 0.0, 'rms_var': 0.0,
        
        # Spectral (6)
        'spectral_centroid_mean': 0.0, 'spectral_centroid_var': 0.0,
        'spectral_bandwidth_mean': 0.0, 'spectral_bandwidth_var': 0.0,
        'rolloff_mean': 0.0, 'rolloff_var': 0.0,
        
        # ZCR (2)
        'zero_crossing_rate_mean': 0.0, 'zero_crossing_rate_var': 0.0,
        
        # Harmony/Percussive (4)
        'harmony_mean': 0.0, 'harmony_var': 0.0,
        'perceptr_mean': 0.0, 'perceptr_var': 0.0,
        
        # Tempo (1)
        'tempo': 0.0,
        
        # MFCCs (40 = 20 means + 20 variances)
        **{f'mfcc{i}_mean': 0.0 for i in range(1, 21)},
        **{f'mfcc{i}_var': 0.0 for i in range(1, 21)}
    }

    # 1. Chroma STFT
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.update({
        'chroma_stft_mean': float(np.mean(chroma)),
        'chroma_stft_var': float(np.var(chroma))
    })

    # 2. RMS Energy
    rms = librosa.feature.rms(y=y)
    features.update({
        'rms_mean': float(np.mean(rms)),
        'rms_var': float(np.var(rms))
    })

    # 3. Spectral Features
    features.update({
        'spectral_centroid_mean': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        'spectral_centroid_var': float(np.var(librosa.feature.spectral_centroid(y=y, sr=sr))),
        'spectral_bandwidth_mean': float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
        'spectral_bandwidth_var': float(np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
        'rolloff_mean': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
        'rolloff_var': float(np.var(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    })

    # 4. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y)
    features.update({
        'zero_crossing_rate_mean': float(np.mean(zcr)),
        'zero_crossing_rate_var': float(np.var(zcr))
    })

    # 5. Harmony/Percussive
    y_harmonic = librosa.effects.harmonic(y)
    y_percussive = librosa.effects.percussive(y)
    features.update({
        'harmony_mean': float(np.mean(y_harmonic)),
        'harmony_var': float(np.var(y_harmonic)),
        'perceptr_mean': float(np.mean(y_percussive)),
        'perceptr_var': float(np.var(y_percussive))
    })

    # 6. Tempo
    features['tempo'] = float(librosa.beat.tempo(y=y, sr=sr)[0])

    # 7. MFCCs (20 means + 20 variances)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f'mfcc{i+1}_mean'] = float(np.mean(mfcc[i]))
        features[f'mfcc{i+1}_var'] = float(np.var(mfcc[i]))  # Now includes mfcc20_var

    print(f"✅ Features extracted: {len(features)} (Should be 58)")
    return features

@app.route('/', methods=['GET', 'HEAD'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST','post'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if not file:
        return redirect(request.url)

    try:
        # File handling
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Feature extraction
        raw_features = extract_features(filepath)
        
        # Define feature order (58 features)
        feature_order = [
            'length',
            'chroma_stft_mean', 'chroma_stft_var',
            'rms_mean', 'rms_var',
            'spectral_centroid_mean', 'spectral_centroid_var',
            'spectral_bandwidth_mean', 'spectral_bandwidth_var',
            'rolloff_mean', 'rolloff_var',
            'zero_crossing_rate_mean', 'zero_crossing_rate_var',
            'harmony_mean', 'harmony_var',
            'perceptr_mean', 'perceptr_var',
            'tempo',
            'mfcc1_mean', 'mfcc1_var',
            'mfcc2_mean', 'mfcc2_var',
            'mfcc3_mean', 'mfcc3_var',
            'mfcc4_mean', 'mfcc4_var',
            'mfcc5_mean', 'mfcc5_var',
            'mfcc6_mean', 'mfcc6_var',
            'mfcc7_mean', 'mfcc7_var',
            'mfcc8_mean', 'mfcc8_var',
            'mfcc9_mean', 'mfcc9_var',
            'mfcc10_mean', 'mfcc10_var',
            'mfcc11_mean', 'mfcc11_var',
            'mfcc12_mean', 'mfcc12_var',
            'mfcc13_mean', 'mfcc13_var',
            'mfcc14_mean', 'mfcc14_var',
            'mfcc15_mean', 'mfcc15_var',
            'mfcc16_mean', 'mfcc16_var',
            'mfcc17_mean', 'mfcc17_var',
            'mfcc18_mean', 'mfcc18_var',
            'mfcc19_mean', 'mfcc19_var',
            'mfcc20_mean', 'mfcc20_var'  # Now included
        ]
        
        # Create ordered features
        ordered_features = [raw_features[feature] for feature in feature_order]
        print(f"✅ Ordered features count: {len(ordered_features)}")
        
        # Verify feature count
        if hasattr(scaler, 'n_features_in_'):
            print(f"Model expects: {scaler.n_features_in_} features")
            if len(ordered_features) != scaler.n_features_in_:
                missing = set(scaler.feature_names_in_) - set(feature_order)
                return f"❌ Feature mismatch! Model expects {scaler.n_features_in_}, got {len(ordered_features)}. Missing: {missing}"
        
        # Make prediction
        features_scaled = scaler.transform([ordered_features])
        prediction = model.predict(features_scaled)[0]
        genre = genre_mapping.get(prediction, "unknown")
        
        return render_template('results.html', 
                            filename=filename,
                            genre=genre,
                            features=raw_features)
    
    except Exception as e:
        error_msg = f"❌ Prediction failed: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg

if __name__ == '__main__':
    load_models()
    app.run(debug=True)