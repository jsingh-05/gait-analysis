# 🦵 Pathological Gait & Activity Recognition using Deep Learning

## 📌 Project Overview

This project focuses on Human Activity Recognition (HAR) combined with Gait Anomaly Detection** to monitor walking patterns and identify possible mobility issues.

It uses deep learning models to:

* Classify human activities (Walking, Sitting, Running, etc.)
* Detect abnormal gait patterns using reconstruction error

---

## 🧠 Models Used

* CNN + BiLSTM → for temporal feature learning
* **Transformer Model** → for sequence modeling (Best Performance)
* Autoencoder → for anomaly (gait) detection

---

## 📊 Dataset

* Sensor-based human activity dataset (e.g., HuGaDB)
* Time-series signals from wearable sensors

---

## ⚙️ Features

* Activity Classification (multi-class)
* Gait Anomaly Detection (Normal / Abnormal)
* Real-time prediction demo
* Model comparison (CNN-BiLSTM vs Transformer)
* Excel-based evaluation reports

---

## 📈 Model Performance

| Model           | Accuracy   | Precision  | Recall     | F1 Score   |
| --------------- | ---------- | ---------- | ---------- | ---------- |
| CNN + BiLSTM    | 89.67%     | 93.13%     | 89.67%     | 90.46%     |
| **Transformer** | 91.89%     | 93.16%     | 91.89%     | 92.24%     |

🏆 **Best Model: Transformer**

---

## 🧪 Sample Prediction Output

```
🦵 GAIT ANALYSIS RESULT
Activity     : Walking
Description  : Normal walking on flat surface
Confidence   : 100.0%
Gait Status  : NORMAL ✅
Recon Error  : 0.581558
Condition    : Healthy Gait
```

---

## 🔍 Batch Prediction Results

```
[1] True: Walking   | Pred: Walking   ✅ | Conf: 100% | NORMAL ✅
[2] True: Sitting   | Pred: Sitting   ✅ | Conf: 100% | NORMAL ✅
[3] True: Bicycling | Pred: Bicycling ✅ | Conf: 100% | NORMAL ✅
```

---

## 📊 Confusion Matrix

Saved in:

```
result/confusion_matrix/
```

---

## 📁 Project Structure

```
gait_project/
│
├── train_classifier.py
├── train_anomaly.py
├── predict.py
├── evaluate.py
├── requirements.txt
├── result/
└── README.md
```

---

## ⚠️ Dataset & Models

Due to large file size, dataset and trained models are hosted externally:

👉 Download here:
([Add your Google Drive link here](https://drive.google.com/drive/folders/1_psjZIEIoH498C_S3VZHqPHhJxlMdhly?usp=drive_link))

Steps:

Download files
Extract into project folder
Run project

<img width="310" height="392" alt="Screenshot 2026-04-25 at 1 09 47 PM" src="https://github.com/user-attachments/assets/07baaff0-9d2c-499d-b995-007c73bec8ac" />

---



## 🚀 How to Run

```bash
git clone https://github.com/Sahil9914/gait_project.git
cd gait_project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python predict.py
```

---

## 🎯 Key Highlights

* Achieved **~92% accuracy**
* Combined classification + anomaly detection
* Real-world applicable system for healthcare monitoring

---


## 👨‍💻 Author

Sahil Chalotra
