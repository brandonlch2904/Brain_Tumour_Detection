# Brain Tumour Detection

This repository demonstrates how to build, train, and evaluate a brain tumour detection model on MRI scans using PyTorch’s Faster R-CNN, with data managed via Roboflow and hyperparameters tuned using Optuna.

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Dataset](#dataset)  
- [Usage](#usage)  
- [Notebook Structure](#notebook-structure)  
- [Model Architecture](#model-architecture)  
- [Hyperparameter Optimization](#hyperparameter-optimization)  
- [Evaluation Metrics](#evaluation-metrics)  
- [Results](#results)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Overview

`main_code.ipynb` walks through:

1. **Environment setup** (installing PyTorch, Roboflow, Optuna, etc.)  
2. **Data import & preprocessing** via the Roboflow API  
3. **Custom PyTorch Dataset** for loading images + annotations  
4. **Model definition** using `fasterrcnn_resnet50_fpn`  
5. **Training & validation loops** with logging  
6. **Hyperparameter tuning** with Optuna  
7. **Performance evaluation** (confusion matrix, PR/ROC curves, mAP)  
8. **Result visualization** of metrics and sample detections  

---

## Features

- 🛠️ **Easy setup**: single notebook installs all dependencies  
- 📦 **Data handling**: download & prepare COCO-formatted data via Roboflow  
- 🔍 **Detection model**: Faster R-CNN with ResNet-50 FPN backbone  
- 🎯 **Customization**: swap in your own dataset or augmentations  
- ⚙️ **Tuning**: Optuna-based search for LR, weight decay, batch size, etc.  
- 📊 **Metrics**: classification report, confusion matrix, precision-recall & ROC curves, mAP@50  

---

## Requirements

- Python 3.8+  
- PyTorch  
- torchvision, torchaudio  
- roboflow  
- optuna  
- scikit-learn, pandas, numpy  
- matplotlib, Pillow, opencv-python-headless  

---

## Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/brandonlch2904/Brain_Tumour_Detection.git
cd Brain_Tumour_Detection
pip install torch torchvision torchaudio roboflow optuna scikit-learn pandas matplotlib pillow opencv-python-headless
