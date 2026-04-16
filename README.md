# 🤟 AI Sign Language Decoder

A full-stack, production-ready AI-powered **Sign Language Decoder** system built with Python. Uses your device's live camera feed to detect and translate sign language gestures into text and speech in **real time**.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-red?logo=google)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Features

| Feature | Description |
|---|---|
| **Real-Time Detection** | Live webcam gesture recognition at ≥20 FPS |
| **Hand Tracking** | MediaPipe-powered 21-keypoint landmark detection |
| **ML Prediction** | Neural network classifies A–Z, 0–9, and common words |
| **Sentence Builder** | Auto-builds sentences from detected signs |
| **Text-to-Speech** | Convert output text to voice with one click |
| **Confidence Scores** | Real-time accuracy display with color coding |
| **Smooth Predictions** | Majority voting eliminates flickering |
| **Modern UI** | Glassmorphism dark theme with animations |
| **Keyboard Shortcuts** | Full keyboard control for power users |

---

## 🏗️ Project Structure

```
sign-language-decoder/
├── app.py                  # Flask server entry point
├── config.py               # All configuration settings
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── model/
│   ├── train.py            # Data collection & model training
│   ├── predict.py          # Gesture prediction engine
│   └── sign_language_model.h5  # Trained model (auto-generated)
│
├── utils/
│   ├── hand_tracker.py     # MediaPipe hand detection
│   ├── preprocessing.py    # Feature extraction & normalization
│   ├── smoothing.py        # Prediction smoothing (majority voting)
│   └── helpers.py          # Logging, JSON, utilities
│
├── api/
│   └── routes.py           # Flask API endpoints
│
├── static/
│   ├── style.css           # Premium glassmorphism CSS
│   └── script.js           # Frontend logic
│
├── templates/
│   └── index.html          # Web interface
│
└── data/
    └── dataset_placeholder/
```

---

## ⚡ Quick Start

### 1. Clone & Navigate

```bash
cd sign-language-decoder
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

The app will:
1. Auto-generate a demo model (first run only, ~30 seconds)
2. Start the Flask server
3. Open at **http://127.0.0.1:5000**

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main web interface |
| `/api/video_feed` | GET | MJPEG video stream |
| `/api/start_camera` | POST | Start webcam capture |
| `/api/stop_camera` | POST | Stop webcam capture |
| `/api/predict` | GET | Get current prediction & sentence |
| `/api/reset` | POST | Clear sentence |
| `/api/speak` | POST | Text-to-speech output |
| `/api/status` | GET | System status |
| `/api/add_char` | POST | Manually add character |
| `/api/backspace` | POST | Delete last character |

---

## 🎮 Keyboard Shortcuts

| Key | Action |
|---|---|
| `S` | Start camera |
| `Q` | Stop camera |
| `R` | Reset/clear text |
| `Space` | Add space |
| `Enter` | Speak output |
| `Backspace` | Delete last character |

---

## 🤖 Training Your Own Model

### Collect Data

```bash
python model/train.py --collect --samples 100
```
- Shows each gesture class and records hand landmarks via webcam
- Press `S` to start collecting each class, `Q` to quit

### Train Model

```bash
python model/train.py --train --epochs 50
```

### Generate Demo Model (Quick Start)

```bash
python model/train.py --demo
```

---

## 📋 Supported Gestures

- **Alphabets**: A–Z
- **Numbers**: 0–9
- **Common Words**: HELLO, THANKS, YES, NO, PLEASE, SORRY, HELP, LOVE
- **Special**: SPACE, DELETE

---

## 🔧 Configuration

Edit `config.py` to customize:

- Camera resolution and FPS
- Confidence threshold
- Smoothing window size
- Flask host and port
- MediaPipe detection parameters

---

## 🧪 Tech Stack

| Component | Technology |
|---|---|
| Backend | Flask 3.0 |
| Computer Vision | OpenCV 4.9 |
| Hand Detection | MediaPipe Hands |
| ML Framework | TensorFlow / Keras |
| Text-to-Speech | pyttsx3 |
| Frontend | HTML5, CSS3, Vanilla JS |
| Architecture | REST API + MJPEG streaming |

---

## 🚀 Future Scope

- Multi-language translation support
- LSTM model for sequence/motion gestures
- Mobile responsive + PWA
- WebRTC for smoother streaming
- Gesture dataset collection tool
- Accuracy & performance dashboard
- Docker deployment
- AR/VR integration

---

## 📄 License

This project is licensed under the MIT License.

---

**Built with ❤️ using AI & Computer Vision**
