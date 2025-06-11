# 😴 Driver Drowsiness Detection with Yawn Detection

A real-time computer vision project that detects **driver drowsiness** based on **eye closure** and **yawn detection** using webcam video stream. If drowsiness or yawning is detected, an alert sound is played to wake the driver.

![demo](https://github.com/your-username/your-repo-name/assets/demo.gif) <!-- You can add a demo GIF or image here -->

---

## 🔧 Tech Stack

- **Python 3.x**
- **OpenCV** – for video capture and image processing
- **Dlib** – for face and facial landmark detection
- **Imutils** – for convenient landmark handling
- **Scipy** – for distance calculation
- **Pygame** – for playing alarm sound

---

## 📁 Features

- Detects if eyes are closed using **Eye Aspect Ratio (EAR)**
- Detects yawning using **Mouth Aspect Ratio (MAR)**
- Displays live EAR and MAR values on the screen
- Sounds an alarm if drowsiness or yawning is detected
- Highlights face features using landmark detection
- Real-time alerts on video feed:
  - `"Eyes Closed"`
  - `"Yawning"`
  - `"Drowsiness Alert!"`

---

## 🧠 How It Works

### Eye Aspect Ratio (EAR)
EAR is computed using 6 facial landmarks for each eye:
- When the eyes are open, EAR is high
- When eyes close, EAR drops below a threshold

### Mouth Aspect Ratio (MAR)
MAR is computed using 20 facial landmarks around the mouth:
- When mouth opens wide (yawning), MAR rises above a threshold

---

## 📦 Installation

### 1. Clone the Repository
```
git clone https://github.com/sohail2251/RealTime_Drowsiness.git
cd driver-drowsiness-detector
```

### 2.Create Virtual Environment
```
python3 -m venv venv
source venv/bin/activate  # for MAC
source venv\Scripts\activate # for Windows
```

### 3.Download Shape Predictor
```
Download shape_predictor_68_face_landmarks.dat and place it in the project folder.
```

## Run the Drowsiness Detection Script
```
python detect_drowsiness.py
```
---

## 🔊 Alarm Sound
Make sure to add an alert sound file (e.g., aleart.aiff) in the same directory as your script.

You can use any sound format supported by Pygame:
  ```
  pygame.mixer.music.load("aleart.aiff")
```
## Dependencies
  ```
  pip install opencv-python dlib imutils pygame scipy
```



