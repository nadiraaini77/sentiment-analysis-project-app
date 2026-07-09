# 📱 MyXL Sentiment Analysis

Aplikasi web untuk melakukan **analisis sentimen ulasan pengguna MyXL** menggunakan pendekatan **Machine Learning Classification**, dibangun dengan **Streamlit**.

🔗 **Live Demo:** (https://sentiment-analysis-project-app-myxl-202607.streamlit.app/)

---

## 📌 Tentang Project

Project ini bertujuan untuk menganalisis sentimen pengguna terhadap layanan MyXL berdasarkan teks ulasan pengguna.

Dengan pendekatan **Supervised Machine Learning (Classification)**, ulasan pengguna diklasifikasikan ke dalam tiga kategori sentimen:

* **Positive**
* **Neutral**
* **Negative**

Project mencakup proses text preprocessing, feature extraction menggunakan TF-IDF, pembangunan dan perbandingan beberapa model machine learning, evaluasi model terbaik, kalibrasi probabilitas, hingga deployment dalam bentuk aplikasi web interaktif.

---

## 📈 Model Information

Beberapa algoritma klasifikasi yang dibandingkan dalam project ini:

* Multinomial Naive Bayes
* Decision Tree
* Random Forest
* AdaBoost
* Logistic Regression
* Support Vector Machine (LinearSVC)

Berdasarkan hasil perbandingan model, **Support Vector Machine (SVM) dengan LinearSVC** menghasilkan accuracy tertinggi.

| Model | Accuracy |
|-------|---------:|
| SVM (LinearSVC) | 78.73% |
| Logistic Regression | 77.69% |

Model SVM kemudian dikalibrasi menggunakan **CalibratedClassifierCV** agar dapat menghasilkan estimasi probabilitas yang digunakan sebagai **confidence score** pada aplikasi.

---

## 🎯 Sentiment Classification

### 1. Positive 😊

Ulasan yang menunjukkan kepuasan atau pengalaman positif pengguna terhadap layanan MyXL.

**Contoh konteks:**
- Kualitas jaringan yang baik
- Aplikasi berjalan lancar
- Pelayanan memuaskan
- Pengalaman pengguna yang positif

---

### 2. Neutral 😐

Ulasan yang tidak menunjukkan sentimen positif atau negatif secara dominan.

**Contoh konteks:**
- Pernyataan informatif
- Pertanyaan mengenai layanan
- Ulasan dengan sentimen yang ambigu
- Pernyataan tanpa ekspresi kepuasan atau keluhan yang kuat

---

### 3. Negative 😞

Ulasan yang menunjukkan ketidakpuasan atau pengalaman negatif pengguna terhadap layanan MyXL.

**Contoh konteks:**
- Jaringan lambat atau tidak stabil
- Error pada aplikasi
- Kendala transaksi atau pembelian paket
- Keluhan terhadap kualitas layanan

---

## 🗂️ Struktur Project

```bash
/
├── app.py                         # Main Streamlit Application
├── requirements.txt               # Dependencies
├── svm_calibrated_model.pkl       # Calibrated SVM Model
├── tfidf.pkl                      # TF-IDF Vectorizer
└── README.md
```

---

## 🔄 Workflow Project

```text
Google Play Store Reviews
              ↓
         Data Cleaning
              ↓
         Case Folding
              ↓
          Tokenization
              ↓
      Slang Normalization
              ↓
       Stopword Removal
              ↓
           Stemming
              ↓
          Split Data
              ↓
            TF-IDF
              ↓
        Model Training
              ↓
       Model Comparison
              ↓
        Model Evaluation
              ↓
        SVM Calibration
              ↓
      Streamlit Deployment
```

---

## ⚙️ Text Preprocessing

Tahapan preprocessing yang digunakan dalam project ini:

| Tahap | Description |
|-------|-------------|
| Cleaning | Menghapus URL, mention, angka, emoji, dan tanda baca |
| Case Folding | Mengubah seluruh teks menjadi lowercase |
| Tokenization | Memecah kalimat menjadi token atau kata |
| Slang Normalization | Mengubah kata tidak baku menjadi bentuk yang telah dinormalisasi |
| Stopword Removal | Menghapus kata-kata yang tidak memberikan informasi penting |
| Stemming | Mengubah kata menjadi bentuk dasarnya |

---

## 🔠 Feature Extraction

Project menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)** untuk mengubah data teks menjadi representasi numerik yang dapat diproses oleh model machine learning.

TF-IDF memberikan bobot pada kata berdasarkan tingkat kepentingannya dalam dokumen dan keseluruhan corpus.

Jumlah maksimum fitur yang digunakan:

```text
5000 features
```

---

## 🤖 Model Evaluation

Model terbaik dievaluasi menggunakan metrik:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Hasil evaluasi Calibrated SVM:

| Metric | Score |
|--------|------:|
| Accuracy | 0.79 |
| Macro Precision | 0.76 |
| Macro Recall | 0.68 |
| Macro F1-Score | 0.71 |
| Weighted Precision | 0.78 |
| Weighted Recall | 0.79 |
| Weighted F1-Score | 0.78 |

Hasil evaluasi menunjukkan bahwa model memiliki performa terbaik dalam mengenali sentimen **negative**, sementara klasifikasi sentimen **neutral** dan **positive** masih lebih menantang.

---

## 📊 Classification Performance

| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|----------:|-------:|---------:|--------:|
| Negative | 0.81 | 0.90 | 0.85 | 617 |
| Neutral | 0.72 | 0.58 | 0.65 | 261 |
| Positive | 0.76 | 0.54 | 0.63 | 81 |

Dataset testing terdiri dari **959 ulasan**.

---

## 🎯 Model Calibration & Confidence Score

Model terbaik menggunakan **LinearSVC**, yang secara default tidak menyediakan method `predict_proba()`.

Untuk menghasilkan estimasi probabilitas, model dikalibrasi menggunakan **CalibratedClassifierCV**.

Hasil probabilitas digunakan untuk menampilkan **confidence score**, yaitu tingkat keyakinan model terhadap kelas sentimen yang diprediksi.

Contoh output aplikasi:

```text
Input:
gilaaa lemot parah

Prediction:
Negative 😞

Confidence Score:
95.81%
```

Confidence score menunjukkan tingkat keyakinan model terhadap hasil prediksi dan bukan ukuran kepastian absolut bahwa prediksi tersebut benar.

---

## 🚀 Cara Menjalankan Lokal

```bash
# 1. Clone repository
git clone https://github.com/nadiraaini77/sentiment-analysis-project-app.git

# 2. Masuk ke folder project
cd sentiment-analysis-project-app

# 3. Buat virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 4. Install dependency
pip install -r requirements.txt

# 5. Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan berjalan di:

```text
http://localhost:8501
```

---

## 🌐 Deployment

Aplikasi dideploy menggunakan **Streamlit Community Cloud**.

Setiap perubahan pada branch utama repository dapat digunakan untuk memperbarui aplikasi yang telah dideploy.

🔗 Live App:

```text
Tambahkan URL aplikasi setelah deployment berhasil.
```

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Pandas
* NLTK
* Sastrawi
* Scikit-Learn
* Joblib
* TF-IDF Vectorizer
* Support Vector Machine (SVM)

---

## 📊 Dataset

Dataset yang digunakan berasal dari ulasan pengguna aplikasi MyXL.

Data ulasan melalui proses preprocessing sebelum digunakan dalam pembangunan model klasifikasi sentimen.

### Target Analisis

Melakukan klasifikasi sentimen untuk:

- Mengidentifikasi sentimen pengguna terhadap layanan MyXL.
- Mengelompokkan ulasan menjadi sentimen Positive, Neutral, dan Negative.
- Membandingkan performa beberapa algoritma machine learning.
- Mengimplementasikan model terbaik dalam aplikasi web interaktif.
- Menampilkan hasil prediksi beserta confidence score.

---

## ⚠️ Catatan

* Model menggunakan pendekatan supervised learning untuk melakukan klasifikasi sentimen.
* Model terbaik menggunakan LinearSVC yang dikalibrasi agar dapat menghasilkan estimasi probabilitas.
* Confidence score merupakan estimasi keyakinan model terhadap hasil prediksi dan bukan jaminan kebenaran prediksi.
* Performa model dapat dipengaruhi oleh distribusi kelas, kualitas preprocessing, dan karakteristik data training.
* Project dibuat untuk tujuan pembelajaran dan analisis data.
