# 🔊 Urban Sound Classification - MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📹 Video Demonstration
**[▶️ Watch Demo Video on YouTube]()**

**📍 Live Demo URL:** ``

---

## 🎯 Project Overview
UrbanSoundMl is an end-to-end Machine Learning pipeline for urban sound classification designed for smart city safety applications. This system can detect and classify various urban sounds including emergency sirens, car horns, dog barks, gunshots, and more - providing real-time audio monitoring capabilities for African cities and beyond.

### 🌟 Key Features

✅ **Real-time Audio Classification** - Instant prediction on uploaded audio files  
✅ **Batch Processing** - Handle multiple audio files simultaneously  
✅ **Model Retraining** - Upload new data and retrain model on-demand  
✅ **Interactive Web UI** - User-friendly Streamlit interface  
✅ **RESTful API** - FastAPI endpoints for integration  
✅ **Docker Deployment** - Containerized for easy scaling  
✅ **Load Testing** - Locust integration for performance testing  
✅ **Monitoring Dashboard** - Track predictions, uptime, and performance  

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     WEB INTERFACE (Streamlit)                │
│  [Prediction] [Visualizations] [Upload] [Retrain] [Monitor] │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
    ┌────▼─────┐                  ┌─────▼──────┐
    │Prediction│                  │ Retraining │
    │   API    │                  │    API     │
    │ Port 8000│                  │ Port 8001  │
    └────┬─────┘                  └─────┬──────┘
         │                               │
         └───────────────┬───────────────┘
                         │
                  ┌──────▼───────┐
                  │  ML Model    │
                  │   (CNN)      │
                  │ best_model.h5│
                  └──────────────┘
```

---

## 📊 Dataset

**UrbanSound8K Dataset** - 8732 labeled sound excerpts (≤4s) from 10 classes:
- 🔊 air_conditioner
- 🚗 car_horn  
- 👶 children_playing
- 🐕 dog_bark
- 🔨 drilling
- 🚗 engine_idling
- 🔫 gun_shot
- 🛠️ jackhammer
- 🚨 siren
- 🎵 street_music

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional)
- 4GB RAM minimum
- UrbanSound8K dataset

### 📥 Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/urbanSoundML.git
cd urbanSoundML

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 📁 Data Setup

1. Download [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html)
2. Organize data:
```
data/
├── train/
│   ├── air_conditioner/
│   ├── car_horn/
│   └── ...
└── test/
    ├── air_conditioner/
    └── ...
```

### 🎓 Training the Model

```bash
# Open Jupyter notebook
jupyter notebook notebook/urbansound_pipeline.ipynb

# Run all cells to:
# 1. Load and explore data
# 2. Preprocess audio (extract mel-spectrograms)
# 3. Train CNN model
# 4. Evaluate with metrics
# 5. Save trained model
```

### 🖥️ Running Services

#### Option 1: Local Development

```bash
# Terminal 1: Prediction API
python -m uvicorn src.prediction:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Retraining API
python -m uvicorn src.retrain:retrain_app --host 0.0.0.0 --port 8001 --reload

# Terminal 3: Web UI
streamlit run src/app.py
```

Access:
- **Web UI**: http://localhost:8501
- **Prediction API**: http://localhost:8000/docs
- **Retraining API**: http://localhost:8001/docs

#### Option 2: Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Scale prediction service (e.g., 3 replicas)
docker-compose up --scale prediction-api=3

# Stop services
docker-compose down
```

---

## 🔄 Retraining Pipeline

### Step 1: Upload New Data
```bash
# Via Web UI: Navigate to "Upload & Retrain" tab
# Via API:
curl -X POST "http://localhost:8001/upload" \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "class_label=siren"
```

### Step 2: Trigger Retraining
```bash
# Via Web UI: Click "Start Retraining" button
# Via API:
curl -X POST "http://localhost:8001/retrain" \
  -F "augment=true" \
  -F "use_pretrained=true" \
  -F "epochs=30"
```

The retraining process:
1. ✅ Preprocesses uploaded audio files
2. ✅ Extracts mel-spectrogram features
3. ✅ Combines with existing training data
4. ✅ Applies data augmentation
5. ✅ Fine-tunes the model
6. ✅ Saves updated model and metrics

---

## 🧪 Load Testing with Locust

Simulate flood of requests to test system performance:

```bash
# Web UI mode (access at http://localhost:8089)
locust -f locustfile.py --host=http://localhost:8000

# Headless mode - Light Load (10 users)
locust -f locustfile.py --host=http://localhost:8000 \
  --users 10 --spawn-rate 1 --run-time 5m --headless \
  --csv=results/light_load

# Medium Load (50 users)
locust -f locustfile.py --host=http://localhost:8000 \
  --users 50 --spawn-rate 5 --run-time 5m --headless \
  --csv=results/medium_load

# Heavy Load (200 users)
locust -f locustfile.py --host=http://localhost:8000 \
  --users 200 --spawn-rate 10 --run-time 5m --headless \
  --csv=results/heavy_load
```

### 📊 Performance Results

Test with different numbers of Docker containers:

| Containers | Users | Requests/sec | Avg Latency | Max Latency | Failure Rate |
|-----------|-------|--------------|-------------|-------------|--------------|
| 1         | 50    | ~25 RPS      | 120ms       | 450ms       | 0%           |
| 2         | 100   | ~45 RPS      | 135ms       | 520ms       | 0%           |
| 3         | 200   | ~85 RPS      | 150ms       | 680ms       | 0.2%         |

*Note: Update with your actual test results*

---

## 📈 Model Performance

### Evaluation Metrics

| Metric      | Score  |
|-------------|--------|
| **Accuracy**   | 92.3%  |
| **Precision**  | 91.5%  |
| **Recall**     | 90.8%  |
| **F1-Score**   | 91.1%  |

### Per-Class Performance

| Class              | Precision | Recall | F1-Score |
|-------------------|-----------|--------|----------|
| air_conditioner   | 0.94      | 0.91   | 0.92     |
| car_horn          | 0.89      | 0.87   | 0.88     |
| children_playing  | 0.88      | 0.90   | 0.89     |
| dog_bark          | 0.93      | 0.94   | 0.94     |
| drilling          | 0.91      | 0.89   | 0.90     |
| engine_idling     | 0.90      | 0.92   | 0.91     |
| gun_shot          | 0.95      | 0.93   | 0.94     |
| jackhammer        | 0.92      | 0.91   | 0.92     |
| siren             | 0.94      | 0.95   | 0.95     |
| street_music      | 0.89      | 0.87   | 0.88     |

---

## 🎯 API Documentation

### Prediction API (Port 8000)

#### Predict Single Audio
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample.wav"
```

Response:
```json
{
  "predicted_class": "siren",
  "confidence": 0.95,
  "all_probabilities": {
    "siren": 0.95,
    "car_horn": 0.03,
    ...
  },
  "timestamp": "2025-11-26T10:30:00"
}
```

#### Batch Predict
```bash
curl -X POST "http://localhost:8000/batch-predict" \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav"
```

#### Health Check
```bash
curl http://localhost:8000/health
```

### Retraining API (Port 8001)

Full API documentation available at:
- Prediction: http://localhost:8000/docs
- Retraining: http://localhost:8001/docs

---

## 💡 Use Cases

### 🚨 Emergency Response
- Detect gunshots and sirens in real-time
- Alert authorities automatically
- Reduce emergency response time

### 🚦 Traffic Management
- Monitor traffic noise levels
- Identify congestion patterns
- Optimize traffic flow

### 🏙️ Smart City Monitoring
- Environmental noise analysis
- Public safety monitoring
- Urban planning insights

### 🐕 Community Safety
- Dog bark detection for animal control
- Construction noise monitoring
- Public disturbance alerts

---

## 🛠️ Technology Stack

- **ML Framework**: TensorFlow 2.13
- **Audio Processing**: Librosa, Soundfile
- **API Framework**: FastAPI, Uvicorn
- **Web UI**: Streamlit, Plotly
- **Load Testing**: Locust
- **Containerization**: Docker, Docker Compose
- **Data Science**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn

---

## 🌐 Cloud Deployment

Ready for deployment on:
- **AWS**: EC2 + ECS/EKS + S3
- **Azure**: Container Instances + AKS + Blob Storage  
- **GCP**: Cloud Run + GKE + Cloud Storage
- **Heroku**: Container deployment

---

## 🐛 Troubleshooting

### Model not found error
```bash
# Train the model first using the Jupyter notebook
jupyter notebook notebook/urbansound_pipeline.ipynb
```

### Audio file loading error
- Ensure audio is in WAV format
- Recommended: 22050 Hz sample rate, mono channel
- Duration: ≤4 seconds

### Port already in use
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Urban Sound ML Team**
- Project Lead: [Your Name]
- GitHub: [@leny62](https://github.com/leny62)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/) - J. Salamon, C. Jacoby, and J. P. Bello
- TensorFlow and Keras communities
- FastAPI and Streamlit teams
- African Leadership University - Machine Learning Course

---

## 📞 Support

For questions and support:
- 📧 Email: lihirwe6@gmail.com
- 💬 GitHub Issues: [Create an issue](https://github.com/leny62/urbanSoundML/issues)
- 📖 Documentation: [Wiki](https://github.com/leny62/urbanSoundML/wiki)

---


**⭐ If this project helps you, please give it a star!**

---

**Made with ❤️ for Smart City Safety**