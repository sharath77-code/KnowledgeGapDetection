# 🧠 EduGap AI — Knowledge Gap Detection & Personalized Learning System

An end-to-end intelligent web platform for automated student knowledge gap detection, machine learning performance tracking, deep knowledge tracing (DKT), and hybrid academic recommendations.

---

## 🌟 Key Features

- 📊 **Dynamic Diagnostic Assessment**: 10-question balanced sampling across 10 distinct mathematical skill domains.
- 🤖 **Dual-AI Architecture**:
  - **XGBoost Classifier (`best_model.pkl`)**: Predicts overall student knowledge gap levels (High, Medium, Low).
  - **PyTorch LSTM Deep Knowledge Tracing (`lstm_model.pth`)**: Models student learning trajectories as a sequential time series across 10 assessment steps.
- 🎯 **1-to-1 Skill-Wise Performance Tracking**: Evaluates accuracy percentages per skill and categorizes performance into:
  - 🚨 **Weak Skills (Action Required)** ($< 70\%$ Accuracy)
  - 🌟 **Strong Competencies** ($\ge 70\%$ Accuracy)
- 💡 **Hybrid Learning Recommendation Path**: Automatically ranks student deficits (lowest accuracy first) and serves personalized Khan Academy video streams, master academic notes, and downloadable practice worksheets.
- 📈 **Interactive Glassmorphism Dashboard**: Built with Chart.js donut charts, dynamic skill mastery bar charts, and circular gap index gauges.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.10+, Flask, Jinja2 Templates
- **Machine Learning**: XGBoost, Scikit-Learn, NumPy, Pandas
- **Deep Learning**: PyTorch (LSTM Recurrent Neural Networks)
- **Frontend**: Modern Vanilla JavaScript (ES6+), HTML5, Vanilla CSS3 (Glassmorphism UI Theme)
- **Visualization**: Chart.js 4.x, FontAwesome 6

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.9+ installed
- Git installed

### 2. Clone Repository & Setup Environment
```bash
git clone https://github.com/YOUR_USERNAME/KnowledgeGapDetection.git
cd KnowledgeGapDetection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Application
```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000` to take an assessment!

---

## 📁 Repository Structure

```
KnowledgeGapDetection/
├── app.py                      # Main Flask application server & route handlers
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git untracked files pattern
├── app/
│   ├── static/
│   │   ├── css/                # Custom Glassmorphism styles
│   │   └── js/                 # Quiz interaction, Chart.js & timer logic
│   └── templates/              # Jinja2 HTML views (quiz, result, recommendations)
├── models/
│   ├── best_model.pkl          # Trained XGBoost ML Pipeline model
│   ├── lstm_model.pth          # PyTorch LSTM Deep Knowledge Tracing weights
│   ├── label_encoder.pkl       # Target label encoder
│   └── dl_scaler.pkl           # Feature scaler
├── notebooks/                  # End-to-end Jupyter Notebooks (EDA, ML, PyTorch DKT)
└── recommendation/
    ├── questions.json          # Master 50-question database across 10 skills
    └── resources.csv           # Skill recommendations, Khan Academy links & notes
```

---

## 📜 License
Licensed under the [MIT License](LICENSE).
