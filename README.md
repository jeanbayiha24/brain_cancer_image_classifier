# 🧠 Brain Cancer MRI Image Classifier

This project is a **web-based image classification tool** that allows users to upload brain MRI (Magnetic Resonance Imaging) scans and receive predictions about potential brain tumor types:

- **glioma**
- **meningioma**
- **pituitary**
- **notumor** (no detectable tumor in the MRI)

The app exposes two deep learning models via a Flask interface, with class probabilities, confidence-aware messages, and a small dashboard of global metrics.

---

## 🧬 Model Architectures

- **TensorFlow model — MobileNetV2**
  - Pretrained on ImageNet, used as a frozen feature extractor.
  - Custom classification head with GlobalAveragePooling + dense layers.
  - Input size: \(96 \times 96 \times 3\).

- **PyTorch model — Custom CNN**
  - Convolutional architecture trained from scratch on the brain MRI dataset.
  - 2 convolution + pooling blocks, followed by fully connected layers.
  - Input size: \(224 \times 224 \times 3\).

Both models are trained on the same 4-class dataset (glioma, meningioma, pituitary, notumor) and evaluated on a held-out test set.

---

## 🖥 Application Features

The Flask web app provides:

- **Single-image inference**
  - Upload an MRI, choose a model (PyTorch or TensorFlow), and get a prediction.
  - Class-specific messages for the 4 tumor types and for “no tumor”.
  - Confidence-aware messages when the model is not clearly certain.

- **Class probabilities**
  - Probabilities for all 4 classes displayed as percentage bars.
  - Useful to inspect borderline cases or potential misclassifications.

- **Last 5 predictions**
  - A compact history table (model, prediction, top class, confidence).
  - Helps visualize how the model behaves across several inputs in a row.

- **Metrics dashboard**
  - A `/metrics` page showing:
    - Confusion matrices for both models.
    - Global accuracy and per-class metrics (precision, recall, F1-score).
  - Metrics are generated offline from the test set and saved as PNG/JSON files.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/jeanbayiha24/brain_cancer_image_classifier.git
cd brain_cancer_image_classifier
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# ou
.\.venv\Scripts\activate    # Windows PowerShell
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Generate metrics and confusion matrices

```bash
python generate_metrics.py
```

This will create:

- `static/confusion_torch.png`, `static/confusion_tf.png`
- `static/metrics_torch.json`, `static/metrics_tf.json`

### 5. Run the Flask app

```bash
python app.py
```

Then open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 💾 Models

- `jean_bayiha_model.weights.h5` — TensorFlow MobileNetV2 model.
- `jean_bayiha_model.torch` — PyTorch custom CNN model.

---

## Disclaimer

Models are trained on a specific Kaggle brain MRI dataset. Predictions on images from other sources (different scanners, hospitals, modalities, or web images) may be unreliable and should not be used for clinical decisions.

---

## 👨‍💻 Author

**Jean Bayiha**

- GitHub: [https://github.com/jeanbayiha24](https://github.com/jeanbayiha24)

---

## 📚 References

- MobileNetV2 paper: [https://arxiv.org/pdf/1801.04381](https://arxiv.org/pdf/1801.04381)
