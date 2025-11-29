# 📦 Fruit Classification with CNN & MLP (PyTorch)

A clean, reproducible deep-learning workflow for classifying fruit images (cherry, tomato, strawberry) using **Convolutional Neural Networks (CNN)** and **Multi-Layer Perceptrons (MLP)** in PyTorch.  
This project demonstrates reproducible data processing, modular model design, and standardized training/evaluation flows.

---

## 🚀 Project Objectives

- Implement two deep-learning architectures (MLP vs. CNN) for image classification  
- Design a **modular workflow** separating dataset loading, model definitions, training loops, and evaluation  
- Provide a **reproducible pipeline** compatible with workflow engines   

---


---

## 🧠 Implemented Models

### **1. Multi-Layer Perceptron (MLP)**
- Baseline model using flattened images  
- Optimizer: SGD  
- Epochs: 5  
- Expected: Lower accuracy (no spatial feature extraction capabilities)

### **2. Convolutional Neural Network (CNN)**
- 3 convolutional layers, ReLU, MaxPool  
- Dropout for regularization  
- Optimizers: SGD & Adam  
- Epochs: 10  
- Expected: Stronger performance due to spatial feature learning

---

## 📊 Performance Summary

| Model | Validation Accuracy | Test Accuracy |
|-------|---------------------|---------------|
| **MLP** | ~50% | ~48–52% |
| **CNN** | ~73% | ~74.6% |

CNN significantly outperforms MLP.

---

## 📦 Dataset

Contains 3 classes:

- cherry  
- tomato  
- strawberry  

To use your own dataset:

data/
├── cherry/
├── tomato/
└── strawberry/


Each folder should contain `.jpg` or `.png` images.


---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/fruit-classification-pytorch.git
cd fruit-classification-pytorch
```

## Install dependencies:

```bash
pip install -r requirements.txt
```

# Train CNN

```bash
python src/train.py --model cnn
```

# Train MLP

```bash
python src/train.py --model mlp
```

# Or use the notebook:
```bash
notebooks/train.ipynb
```

# Evaluation
```bash
python src/evaluate.py --model cnn
```

# Or use:
```bash
notebooks/test.ipynb
```

