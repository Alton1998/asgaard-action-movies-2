
# 🎬 SigLIP-Powered Shot Boundary Detection for Action Movies

### Automating the Identification of Action Sequences in Long-Form Films

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![License](https://img.shields.io/github/license/Alton1998/asgaard-action-movies-2)
![LLM](https://img.shields.io/badge/LLM-Enabled-purple)
![Vision-Language](https://img.shields.io/badge/Vision--Language-Model-green)

---

## 📌 Overview

This project automates the detection of **action sequences in long-form movies** by integrating:

- **Shot Boundary Detection (SBD)** using Sigmoid Loss for Language-Image Pretraining (SigLIP)
- **Key Shot Extraction (KSE)**
- **Zero-shot Action Classification** via Vision-Language Models (VLMs)

It significantly outperforms traditional methods like AutoShot, especially on gradual transitions in long-form media. The system enables efficient video annotation and the creation of structured **Action Profiles**—a tool for analyzing action patterns in films.

---

## 🧠 Core Contributions

- 📉 **49% improvement** over AutoShot in SBD accuracy
- 🧠 Uses **SigLIP + cosine similarity** to detect shot transitions without retraining
- 🎯 Optimizes threshold (τ ≈ 0.825) for best results
- 🖼️ Extracts representative **Key Shots** using 3D CNN features
- 🧾 Classifies action scenes using **LLaVA-Interleave** models (0.5B and 7B variants)

---

## 📂 Repository Structure

```
asgaard-action-movies-2/
│
├── shot_boundary_detection.py        # Detects shots using SigLIP and cosine similarity
├── key_shot_generator.py             # Generates key shots from detected shots
├── key_shot_annotator_classifier.py  # Annotates and classifies key shots using a fine-tuned LLaVA model
├── trainer.py                        # Trains the action classification model using "The Wild Bunch (1969)-2-0_75.csv"
├── requirements.txt                  # Dependencies
└── README.md                         # Project documentation

```

---

## ⚙️ Methodology

### 1. Frame Extraction
Extract frames from a video at native resolution.

### 2. Shot Boundary Detection (SBD)
- Extract image features via **SigLIP**
- Compare adjacent frames with **cosine similarity**
- Identify boundaries where similarity < τ

### 3. Key Shot Extraction (KSE)
- Select a **representative frame** per shot (closest to feature vector mean)

### 4. Action Classification
- Use **LLaVA (7B preferred)** with image+prompt input
- Perform zero-shot classification into categories:
  - Fight
  - Heist
  - Escape
  - Rescue
  - Capture
  - Pursuit
  - Speed

---

## 📊 Results

| Method         | Start MAE (s) | End MAE (s) | Accuracy Gain vs AutoShot |
|----------------|---------------|-------------|-----------------------------|
| SigLIP (τ=0.825) | 14            | 14          | +49%                        |
| AutoShot        | 28            | 28          | —                           |

---

## 📁 Datasets

- **SBD Evaluation Dataset** from *The Wild Bunch (1969)*  
  - [Download Shot Boundary Ground Truth](https://drive.google.com/file/d/1WiPqupmqC01gvXg4dfgqHFHf6UBJJ1yL/view)

- **Action Classification Dataset** (manually labeled key shots)
  - [Download Annotation Dataset](https://drive.google.com/file/d/1dvnYdUYhl6zQzJggLgC5r36W0nDtqqzX/view)

---

## 🔧 Setup & Running

### 🖥️ Prerequisites
- Python 3.9+
- CUDA-enabled GPU (24GB VRAM recommended)
- PyTorch, OpenCV, FAISS, HuggingFace Transformers, LLaVA models

### 📦 Installation

```bash
git clone https://github.com/Alton1998/asgaard-action-movies-2.git
cd asgaard-action-movies-2
pip install -r requirements.txt
```

### ▶️ Run Pipeline

```bash
python run_pipeline.py --video_path path_to_video.mp4
```

---

## 💡 Future Work

- Expand support for **gradual transitions** (fade, dissolve)
- Build a larger **action-labeled dataset** for supervised fine-tuning
- Integrate **LLM-assisted annotation feedback loop**

---

## 🙏 Acknowledgements

Special thanks to:
- **Prof. Cor-Paul Bezemer** (Project Advisor)
- **Prof. Li Cheng and Tico Romao** (Domain Knowledge & Dataset)
- **Asgaard Lab**, and other collaborators

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---


