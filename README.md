
# 🚗 Drowsiness Detection System

A real-time **Driver Drowsiness Detection System** built using **Deep Learning, Computer Vision, and Flask**.
The system detects whether a person's eyes are **open or closed** using a CNN model. If the eyes remain closed for more than **2 seconds**, an **alarm sound** is triggered to alert the user.

This project helps reduce accidents caused by driver fatigue by providing an automatic alert system.

---

# 📌 Features

✔ Real-time webcam monitoring
✔ Eye detection using **Haar Cascade**
✔ CNN model for eye state classification
✔ Detects **Open / Closed eyes**
✔ Alarm triggers if eyes remain closed for 2 seconds
✔ Simple **web interface** using HTML and JavaScript
✔ Backend built with **Flask API**

---

# 🧠 Model Architecture

The deep learning model is a **Convolutional Neural Network (CNN)** trained on eye images.

Architecture used:

* Conv2D (32 filters) + MaxPooling
* Conv2D (64 filters) + MaxPooling
* Conv2D (128 filters) + MaxPooling
* Flatten Layer
* Dense Layer (128 neurons)
* Dropout (0.5)
* Output Layer (Sigmoid)

Loss Function:

```
Binary Cross Entropy
```

Optimizer:

```
Adam
```

---

# 🗂 Project Structure

```
Drowsiness-Detection-System
│
├── model
│   └── drowsiness_model.h5
│
├── static
│   └── alarm.wav
│
├── templates
│   └── index.html
│
├── train
│   ├── open_eyes
│   └── closed_eyes
│
├── app.py
├── detect_drowsiness.py
├── script.js
├── requirements.txt
├── runtime.txt
├── Procfile
└── README.md
```

---

# ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* Flask
* HTML
* JavaScript
* CNN (Deep Learning)

---

# 🧑‍💻 How It Works

1. Webcam captures the user's face.
2. **Haar Cascade** detects face and eyes.
3. Eye region is extracted and resized to **64×64**.
4. Image is passed to the **CNN model**.
5. Model predicts:

   * **Open Eye**
   * **Closed Eye**
6. If eyes stay closed for **more than 2 seconds**:

   * Alarm sound is triggered
   * "DROWSY!" warning appears.

---

# 🚀 Running the Project Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/navyasree1385/drowsiness-detection-system.git
cd drowsiness-detection-system
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the Flask app

```bash
python app.py
```

---

### 4️⃣ Open browser

```
http://localhost:5000
```

Allow camera access to start detection.

---

# 🧪 Model Training

The CNN model was trained using **ImageDataGenerator** with augmentation:

* Rescaling
* Rotation
* Zoom
* Width shift
* Height shift
* Horizontal flip

Training settings:

```
Epochs: 15
Batch size: 32
Image size: 64x64
```

The trained model is saved as:

```
drowsiness_model.h5
```

---

# 📊 Future Improvements

* Improve detection accuracy using **larger datasets**
* Use **Eye Aspect Ratio (EAR)** for better detection
* Implement **driver monitoring dashboard**

---

# 👩‍💻 Author

**Navyasree Durga**
B.Tech Student
GITAM University

---
