# 🚉 Smart Railway Surveillance System

An AI-powered surveillance system for enhancing safety and security at railway stations.
This project integrates face recognition, weapon detection, trespassing detection, and a unified system with a Tkinter UI and Flask API alerts.

# 🔍 Features

### 🎯 Face Recognition (DeepFace)

- #### Recognizes missing persons and criminals from a dataset.

- #### Supports specific person recognition by uploading an image at runtime.

### 🔫 Weapon Detection (YOLOv8)

- #### Detects weapons like knives and guns in live video.

- #### Triggers alerts via Flask API.

### 🚷 Trespassing Detection (YOLOv8 Segmentation)

- #### Detects unauthorized entry into railway tracks.

- #### Sends alerts in real-time.

### 📢 Alert System

- #### Flask API provides real-time alerting.


### 🖥️ User Interface (Tkinter)

- #### Displays real-time camera feed.

- #### Shows bounding boxes, detected labels, and alerts.

- #### Allows runtime upload of a target person’s image.

# ⚙️ Installation

### 1️⃣ Clone the repo:
```bash
git clone https://github.com/akhil3002/Smart-Railway-Surveillance-System.git
cd Smart-Railway-Surveillance-System
```
### 2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Flask API:
```bash
python api.py
```
### 4️⃣ Run the Tkinter UI:
```bash
python UI.py
```
# 📹 Working Video
https://www.linkedin.com/feed/update/urn:li:activity:7358815309186678786/
