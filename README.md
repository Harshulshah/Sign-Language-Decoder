# 🤟 Sign Language Decoder

AI-powered real-time sign language recognition system with **136+ signs**, hybrid gesture detection, intelligent word formation, and text-to-speech output.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-red)

## ✨ Features

- **🔤 136+ Signs** — Full ASL alphabet (A-Z), numbers (0-9), and 100+ common words
- **🤖 Hybrid Detection** — Rule-based static classifier + motion trajectory detector
- **💡 Smart Suggestions** — Auto-complete, spell correction, next-word prediction
- **🔊 Text-to-Speech** — Speak sentences using Web Speech API
- **📚 Sign Library** — Browse all signs with descriptions and categories
- **➕ Custom Signs** — Add your own sign definitions
- **🎨 Premium UI** — Dark glassmorphism theme with micro-animations

## 🚀 Quick Start (Local)

```bash
# Clone the repo
git clone https://github.com/Harshulshah/Sign-Language-Decoder.git
cd Sign-Language-Decoder

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Open http://127.0.0.1:5000
```

## ☁️ Live Demo

Deploy on Render.com:

1. Fork this repo
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Select **Docker** runtime
5. Deploy! 🎉

## 📁 Project Structure

```
sign-language-decoder/
├── app.py                  # Flask application entry point
├── config.py               # All configuration settings
├── Dockerfile              # Cloud deployment container
├── render.yaml             # Render.com deployment config
├── requirements.txt        # Python dependencies
│
├── api/
│   └── routes.py           # REST API endpoints (cloud-ready)
│
├── model/
│   ├── hand_rules.py       # Rule-based ASL classifier (A-Z, 0-9)
│   ├── motion_detector.py  # Dynamic gesture detection
│   └── predict.py          # Hybrid prediction pipeline
│
├── utils/
│   ├── hand_tracker.py     # MediaPipe hand detection
│   ├── smoothing.py        # Prediction stabilization (70% majority)
│   └── word_engine.py      # NLP word formation engine
│
├── data/
│   ├── sign_library.json   # 136 sign definitions
│   ├── word_dictionary.json# Dictionary + bigrams
│   └── custom_signs.json   # User-added signs
│
├── templates/
│   └── index.html          # Main UI page
│
└── static/
    ├── style.css           # Premium dark theme
    └── script.js           # Client-side camera + UI logic
```

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask, Python 3.11 |
| Hand Detection | MediaPipe Hands |
| ML Model | Rule-based geometry + TensorFlow |
| Frontend | Vanilla JS, CSS (Glassmorphism) |
| TTS | Web Speech API |
| Deployment | Docker, Render.com |

## 📄 License

MIT License
