# 🎵 Music Genre Classification Web Application

A sophisticated web application that uses machine learning to classify music genres and recommend Spotify playlists. Built with Flask, scikit-learn, and librosa for audio processing.

## 🌟 Features

- **Real-time Music Classification**: Upload and analyze audio files instantly
- **Multi-Genre Support**: Classifies 10 different music genres:
  - Blues
  - Classical
  - Country
  - Disco
  - Hip-Hop
  - Jazz
  - Metal
  - Pop
  - Reggae
  - Rock
- **Advanced Audio Analysis**: Extracts 58 unique audio features including:
  - Chroma STFT
  - RMS Energy
  - Spectral Features
  - Zero Crossing Rate
  - Harmony/Percussive Components
  - MFCCs (Mel-frequency cepstral coefficients)
- **Spotify Integration**: Automatic playlist recommendations based on predicted genre
- **Interactive UI**: Modern, responsive interface with audio playback capabilities
- **Demo Section**: Pre-loaded audio samples for quick testing

## 🛠️ Technology Stack

### Backend
- **Python 3.x**
- **Flask**: Web framework
- **librosa**: Audio processing and feature extraction
- **scikit-learn**: Machine learning models (SVM and KNN)
- **numpy & pandas**: Data processing
- **pickle**: Model serialization

### Frontend
- **HTML5**
- **CSS3**: Custom styling with gradients and animations
- **JavaScript**: Interactive features
- **Jinja2**: Template engine

### Machine Learning
- **Models**: 
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
- **Feature Engineering**: 58 audio features
- **Data Preprocessing**: MinMaxScaler for feature normalization


## 🚀 Installation & Setup

1. Clone the repository
```bash
git clone [your-repository-url]
cd music-genre-classifier
```

2. Create and activate virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
python app.py
```

5. Access the web interface at `http://localhost:5000`

## 📝 Usage

1. **Upload Audio**:
   - Click "Choose File" on the main page
   - Select an audio file (.wav or .mp3)
   - Click "Predict Genre"

2. **View Results**:
   - See predicted genre
   - Explore detailed audio features
   - Get Spotify playlist recommendations

3. **Try Demo**:
   - Use pre-loaded audio samples
   - Test different genres
   - Instant classification

## 🔬 Technical Details

### Feature Extraction
- Audio length
- Chroma STFT (mean & variance)
- RMS Energy (mean & variance)
- Spectral Centroid
- Spectral Bandwidth
- Rolloff
- Zero Crossing Rate
- Harmony/Percussive components
- Tempo
- 20 MFCCs (mean & variance)

### Model Performance
- SVM Accuracy: [Your accuracy score]
- KNN Accuracy: [Your accuracy score]
- Trained on GTZAN dataset


## 🙏 Acknowledgments

- GTZAN Dataset for music genre classification
- Spotify for playlist integration
- Flask and scikit-learn communities

---

**Note**: This project was developed as part of [Your Course/University Details].
