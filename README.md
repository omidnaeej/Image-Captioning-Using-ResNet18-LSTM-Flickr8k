# Flickr8k Image Captioning

## 📌 Project Overview

This project implements an **image captioning model** that generates natural language descriptions for images from the **Flickr8k dataset**.
The approach combines **Convolutional Neural Networks (CNN)** for image feature extraction and **Recurrent Neural Networks (RNN)** with LSTM for sequence generation.

The pipeline covers:

* Dataset download & preprocessing
* Vocabulary creation and text tokenization
* CNN encoder + LSTM decoder model architecture
* Model training with frozen and unfrozen CNN layers
* Caption generation for test images

---

## 📑 Table of Contents

1. [Dataset](#-dataset)
2. [Model Architecture](#-model-architecture)
3. [Installation & Setup](#-installation--setup)
4. [Usage](#-usage)
5. [Results](#-results)
6. [Future Improvements](#-future-improvements)
7. [License](#-license)

---

## 📂 Dataset

We use the **Flickr8k** dataset, which contains:

* **8,000 images** from Flickr
* **5 human-written captions** per image

The dataset is fetched via the **Kaggle API**.

---

## 🏗 Model Architecture

### **1. Encoder (CNN)**

* Pretrained **ResNet-18** (feature extractor)
* Fully connected linear layer
* Batch normalization to reduce overfitting

### **2. Decoder (RNN with LSTM)**

* Embedding layer (word vector representation)
* Dropout (regularization)
* LSTM (sequence modeling)
* Fully connected layer (word prediction)

---

## ⚙ Installation & Setup

### **1. Clone the repository**

```bash
git clone https://github.com/your-username/flickr8k-image-captioning.git
cd flickr8k-image-captioning
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Download the dataset**

* Ensure you have a **Kaggle API token** in `~/.kaggle/kaggle.json`

```bash
pip install kaggle
kaggle datasets download -d adityajn105/flickr8k
unzip flickr8k.zip -d ./data
```

---

## 🚀 Usage

### **Run the Notebook**

Open and execute:

```bash
jupyter notebook Flickr8ImageCaptioning.ipynb
```

Key steps:

1. **Data Preprocessing** – resize images, tokenize captions, build vocabulary
2. **Train Model (Frozen ResNet)** – initial training with encoder frozen
3. **Train Model (Unfrozen ResNet)** – fine-tuning for better accuracy
4. **Predict Captions** – generate captions for random test images

---

## 📊 Results

* Model successfully generates meaningful captions for unseen images
* Fine-tuning (unfreezing ResNet) reduces loss but may require more training time
* Padding and special tokens (`<SOS>`, `<EOS>`, `<PAD>`) ensure better sequence handling

Example Output:

| Image                       | Predicted Caption                       |
| --------------------------- | --------------------------------------- |
| ![Sample](docs/sample1.jpg) | "A group of people standing on a beach" |

---

## 🔮 Future Improvements

* Experiment with deeper CNNs (ResNet50, EfficientNet)
* Add attention mechanism for better word-image alignment
* Train with larger datasets (MS COCO)
* Hyperparameter tuning for better BLEU scores

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
