# 🧠 GNN Framework for Representation Learning in Whole-Slide Histopathology Images

## 📌 Overview

This project presents a Graph Neural Network (GNN)-based framework for automated classification of lung cancer subtypes using Whole Slide Histopathology Images (WSIs). The system focuses on distinguishing between Lung Adenocarcinoma (LUAD) and Lung Squamous Cell Carcinoma (LUSC) by combining advanced image preprocessing, graph-based tissue representation, and self-supervised learning techniques.

Traditional patch-based approaches often ignore spatial relationships between tissue regions, leading to limited diagnostic performance. To address this challenge, the proposed framework constructs tissue graphs from extracted image patches and leverages Graph Neural Networks to learn meaningful representations for accurate cancer subtype classification.

---

## 🎯 Project Objectives

### Objective 1: Informative Patch Selection

Extract diagnostically relevant tissue regions from Whole Slide Images using tissue masking, variance-based filtering, and balanced patch selection techniques.

### Objective 2: Stain Normalization

Reduce inter-slide staining variability through LAB-based stain normalization while preserving important histological structures.

### Objective 3: Graph-Based Representation Learning

Construct patch-level tissue graphs and train a Graph Autoencoder (GAE) followed by a Graph Convolutional Network (GCN) for robust classification of LUAD and LUSC.

---

## 🔧 Methodology

The proposed framework follows a multi-stage pipeline:

1. Whole Slide Image acquisition and preprocessing
2. Tissue mask generation and patch extraction
3. Stain normalization using LAB color space
4. Deep feature extraction using ResNet50
5. K-Nearest Neighbor (KNN) graph construction
6. Self-supervised Graph Autoencoder pretraining
7. Graph Convolutional Network fine-tuning
8. LUAD/LUSC classification and evaluation

---

## 🗂️ Dataset

* Source: TCGA-LUAD and TCGA-LUSC datasets
* Total Whole Slide Images: 42
* Patch Size: 256 × 256 pixels
* Extracted Tissue Patches: Approximately 2,770

---

## 📊 Results

The proposed model achieved excellent classification performance:

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 100%  |
| Precision | 100%  |
| Recall    | 100%  |
| F1-Score  | 100%  |
| AUC-ROC   | 100%  |

Visualization techniques such as PCA, t-SNE, and confusion matrices demonstrated clear separation between LUAD and LUSC tissue representations.

---

## 🔬 Key Contributions

* Efficient preprocessing pipeline for Whole Slide Images
* LAB-based stain normalization for color consistency
* Graph-based tissue representation preserving spatial information
* Self-supervised learning reducing annotation dependency
* Highly accurate lung cancer subtype classification

---

## 🏫 Institution

University of Agriculture, Faisalabad (UAF)
Department of Computer Science
Final Year Project (BI-606)

---

## 📚 Technologies Used

* Python
* PyTorch
* PyTorch Geometric
* OpenSlide
* OpenCV
* ResNet50
* Scikit-Learn
* Matplotlib
