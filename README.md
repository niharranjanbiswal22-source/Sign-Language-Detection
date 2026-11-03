<div align="center">

# 🤟 Sign AI Pro: Real-Time Sign Language to Text & Speech Conversion

### *An Intelligent Deep Learning & Computer Vision Platform for Bridging Communication Barriers*

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Studio-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.2%25-brightgreen?style=for-the-badge&logo=target)](https://github.com/niharranjanbiswal22-source/Sign-Language-Detection)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br/>

[Key Highlights](#-key-engineering-highlights) •
[System Architecture](#-system-architecture) •
[Interactive Web Studio](#-interactive-web-studio) •
[Installation & Quickstart](#-quickstart-guide) •
[Documentation](#-academic-reports--presentation) •
[Contributors](#-developer--author-credits)

---

</div>

## 📌 Executive Summary

Over **70 million deaf individuals worldwide** depend on sign language as their primary mode of daily expression and interaction. Most hearing individuals cannot interpret sign language, creating communication friction in public sectors, healthcare, education, and workplaces.

**Sign AI Pro** is an end-to-end Computer Vision and Deep Learning framework designed to translate **American Sign Language (ASL)** fingerspelling into coherent text and natural vocalized speech in real time. Unlike conventional vision models that fail under varied lighting, skin tones, or complex backgrounds, this project introduces a **background-invariant skeleton normalization pipeline** paired with a **hierarchical Convolutional Neural Network (CNN)** and **geometric heuristic decision tree**, achieving **99%+ translation accuracy** at **30+ FPS**.

---

## ⚡ Key Engineering Highlights

- 🖐️ **Background-Invariant Landmark Normalization**: Raw camera frames are analyzed with MediaPipe to isolate 21 3D hand keypoints, which are dynamically plotted onto a standardized $400 \times 400$ synthetic canvas. This eliminates false positives caused by cluttered backgrounds or lighting variations.
- 🧠 **Hierarchical 8-Group CNN Classification**: To overcome the inter-class confusion inherent in 26-class alphabet models, signs with similar topological profiles are clustered into 8 primary groups, classified via a specialized CNN, and resolved into exact characters using sub-finger joint distance heuristics.
- 🗣️ **Instant Text-to-Speech (TTS) Synthesis**: Integrated audio engine vocalizes translated sentences on-the-fly, enabling natural, two-way conversational flow.
- 💡 **Smart NLP Word Suggestions**: Real-time spell checking and lexical completion via PyEnchant suggest candidate words as the user fingerspells.
- 🌐 **Modern Glassmorphic Web Dashboard**: Complete Flask web suite equipped with camera feed controls, security CAPTCHA authentication, sentence clipboard controls, and timestamped history export.
- ⚡ **Ultra-Low Latency**: Inference executes in under **25ms per frame**, making it fully responsive on standard consumer laptops without demanding high-end GPUs.

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    A[📹 Video Capture] --> B[🖐️ MediaPipe 21 Hand Landmarks]
    B --> C[📐 400x400 Skeleton Normalization]
    C --> D[🧠 8-Group CNN Classifier]
    D --> E[📏 Euclidean Joint Heuristics Engine]
    E --> F[🔤 Character Resolution: A-Z, Space, Backspace]
    F --> G[💡 NLP Word Autocomplete]
    G --> H[🔊 Text-to-Speech Engine]
    G --> I[💾 SQLite Translation History]
```

### 🔬 8-Group Topological Clustering Matrix

| Group ID | Base Characters Cluster | Key Distinguishing Joint Features |
|:---:|:---:|:---|
| **0** | `[A, E, M, N, S, T]` | Thumb cross-over coordinate relative to index/pinky knuckle bases |
| **1** | `[B, D, F, I, K, R, U, V, W]` | Finger extension count & Euclidean fingertip spacing |
| **2** | `[C, O]` | Distance between index tip (Pt 8) and thumb tip (Pt 4) |
| **3** | `[G, H]` | Horizontal finger orientation & joint separation ratio |
| **4** | `[L]` | Perpendicular $90^\circ$ angle between thumb and index finger |
| **5** | `[P, Q, Z]` | Downward index trajectory & thumb positioning vectors |
| **6** | `[X]` | Hooked index curvature relative to intermediate joints |
| **7** | `[J, Y]` | Pinky extension trajectory & lateral wrist movement |

---

## 💻 Interactive Web Studio

The project includes a production-ready Flask Web Studio:
- **Dual Dashboard View**: Live real-time OpenCV MJPEG video stream with visual bounding boxes alongside real-time detected symbol metrics.
- **Sentence Construction Deck**: Space, Backspace, and Clear gestures or UI buttons for seamless communication.
- **Audit & History Export**: User sessions securely record translated interactions into an encrypted SQLite ledger with instant `.txt` file download.
- **Secure Authentication**: User registration and login protected with bcrypt password hashing and dynamically rendered security CAPTCHAs.

---

## 📁 Repository Directory Structure

```plaintext
Sign-Language-Detection/
├── AtoZ_3.1/                      # Curated skeleton dataset (A-Z gestures)
├── docs/                          # Academic documents & deliverables
│   ├── Project_Report.docx        # Full technical project research paper
│   ├── Viva_Presentation.pptx     # Comprehensive slide deck for viva/demos
│   └── Project_Poster.pdf         # Visual academic project poster
├── app.py                         # Production Flask Web Studio & CV pipeline
├── final_pred.py                  # Standalone Tkinter desktop GUI application
├── prediction_wo_gui.py           # Headless CLI prediction script for testing
├── data_collection_final.py       # Landmark-based dataset collection pipeline
├── data_collection_binary.py      # Binary/grayscale sample acquisition script
├── cnn8grps_rad1_model.h5         # Pre-trained 8-group CNN model weights
├── hand_landmarker.task           # MediaPipe vision task package
├── white.jpg                      # Normalization canvas template
├── requirements.txt               # Pinned Python package dependencies
├── .gitignore                     # Production Git exclusion rules
├── LICENSE                        # Open-source MIT License
└── README.md                      # Comprehensive project documentation
```

---

## 🚀 Quickstart Guide

Get the project running locally in **less than 2 minutes**.

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/niharranjanbiswal22-source/Sign-Language-Detection.git
cd Sign-Language-Detection
```

### 2️⃣ Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to:
```text
http://localhost:5000
```

> **Alternative Desktop GUI**: To run the standalone Tkinter desktop interface instead, run:
> ```bash
> python final_pred.py
> ```

---

## 📊 Model Performance & Benchmarking

| Metric | Score | Notes |
|:---|:---:|:---|
| **Controlled Lighting Accuracy** | **99.2%** | Standard indoor lighting, neutral background |
| **Adverse Lighting & Dynamic Backgrounds** | **97.4%** | Mediated by MediaPipe landmark normalization |
| **Inference Time per Frame** | **~24 ms** | Intel i5 / AMD Ryzen 5 CPU (No dedicated GPU required) |
| **FPS Throughput** | **35 - 45 FPS** | Fluid real-time video stream |
| **Memory Footprint** | **< 350 MB** | Lightweight CNN model architecture ($13.5\text{ MB}$) |

---

## 📚 Academic Reports & Presentation

Detailed academic research documentation is included in the [`docs/`](docs/) directory:
- 📄 **Project Report**: [`docs/Project_Report.docx`](docs/Project_Report.docx) - Complete technical paper detailing neural network architecture, data preprocessing, math formulation, and test results.
- 📊 **Viva Presentation**: [`docs/Viva_Presentation.pptx`](docs/Viva_Presentation.pptx) - Ready-to-present PowerPoint slide deck.
- 📜 **Project Poster**: [`docs/Project_Poster.pdf`](docs/Project_Poster.pdf) - High-resolution academic exhibition poster.

---

## 🛠️ Technology Stack

- **Machine Learning & Vision**: TensorFlow, Keras, OpenCV, MediaPipe, cvzone, NumPy
- **Web Backend & Database**: Python, Flask, SQLite3, Werkzeug
- **Frontend & UI**: Bootstrap 5, Modern Glassmorphism CSS, JavaScript (Fetch API)
- **Speech & Audio**: PyTTSx3, Web Speech Synthesis API
- **Natural Language Processing**: PyEnchant

---

## 👨‍💻 Developer & Author Credits

- **Deepak Sethi** — *Lead Algorithm Architect & Full-Stack Developer*
- **Nihar Ranjan Biswal** — *Machine Learning Contributor & Project Maintainer* ([@niharranjanbiswal22-source](https://github.com/niharranjanbiswal22-source))

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - feel free to use, modify, and distribute for personal, educational, or commercial applications.

<div align="center">
⭐ If you find this project valuable, please consider giving it a star on GitHub! ⭐
</div>
