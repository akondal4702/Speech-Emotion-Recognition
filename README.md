# 🎙️ AI-Based Speech Emotion Recognition

*A Wav2Vec2 + Lightweight Neural Classifier SER System*


## 📌 Overview

This project builds a **Speech Emotion Recognition (SER)** model using **Wav2Vec2** embeddings and a **shallow neural classifier** for fast and accurate emotion prediction.
The complete workflow—including preprocessing, feature extraction, training, and evaluation—is implemented in the notebook **DL_SER_Team_17.ipynb**.

The system is designed to be **simple, fast, and suitable for real-time deployment**, following the methodology described in the project report.


---

## 🚀 Key Features

* Audio-only model (unimodal SER)
* Wav2Vec2-base encoder (frozen) for emotion-rich embeddings
* Lightweight 2-layer classifier
* Training + evaluation implemented in the `.ipynb`
* ~95% test accuracy on RAVDESS


---

## 🎧 Dataset: RAVDESS

* 1440 speech files
* 24 speakers (12M/12F)
* 8 emotions: *neutral, calm, happy, sad, angry, fearful, disgust, surprised*
* Clean studio recordings → ideal for SER

---

## 🧠 Model Architecture

### **1. Preprocessing**

* Resample to **16 kHz**
* Mono conversion
* Normalization
* Label extraction based on filename 

### **2. Feature Extraction – Wav2Vec2**

* Model: `facebook/wav2vec2-base`
* Outputs frame-level embeddings
* Mean pooling → **768-dim vector**
* Encoder kept **frozen**

### **3. Classifier Head**

* Linear(768→256)
* ReLU
* Dropout(0.3)
* Linear(256→8 emotion classes)
* Softmax during inference
  (Report section 4.3) 

---

## 🏋️ Training Details

* Loss: **CrossEntropyLoss**
* Optimizer: **AdamW**
* LR = **1e-4** with scheduler
* Batch size = **16**
* Epochs = **10**, early stopping
  (Report section 4.4) 

---

## 📊 Results


* **Accuracy:** 95.31%
* **Macro F1:** 94.92%
* Strong performance on *angry, sad, disgust*
* Most confusion between:

  * neutral ↔ calm
  * happy ↔ surprised

---

## 📦 Installation

```bash
pip install torch torchaudio transformers librosa scikit-learn matplotlib numpy pandas
```

---

## ▶️ Usage

### **Run Training & Evaluation (Notebook)**

Open and run:

```
DL_SER_Team_17.ipynb
```

This notebook handles:
✔ Preprocessing
✔ Wav2Vec2 embedding extraction
✔ Classifier training
✔ Loss/accuracy curves
✔ Confusion matrix
✔ Test predictions

---




