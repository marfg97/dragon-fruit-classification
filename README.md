# Dragon Fruit Quality Classifier 🐉🔍

![Dragon Fruit Examples](https://github.com/marfg97/dragon-fruit-classification/blob/main/Augmented_Dataset/anomaly/Defect_Dragon_Augmented_Data0009.jpg)  
*Fresh vs rotten dragon fruit examples*

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technical Specifications](#technical-specifications)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#1-training)
  - [Inference](#2-inference)
  - [API Deployment](#3-api-deployment)
- [Performance](#performance)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Project Overview
PyTorch-based deep learning system for automated quality control of dragon fruit (pitaya). Classifies fruits as **fresh** or **rotten** with >95% accuracy to assist agricultural quality control processes.

## Key Features
- **Custom CNN Architecture** optimized for fruit imagery
- **Advanced Data Augmentation** pipeline
- **Class Imbalance Solutions**:
  - Weighted random sampling (300 rotten vs 2000 fresh)
  - Focal loss implementation
- **Model Interpretability**:
  - Grad-CAM visualization
  - Confidence threshold tuning
- **Deployment Ready**:
  - SageMaker compatible
  - Flask API template

## Technical Specifications

| Component          | Details                                                                 |
|--------------------|-------------------------------------------------------------------------|
| Framework          | PyTorch 1.12.1                                                         |
| Model Architecture | Resnet18                                       |
| Input Specs        | 256x256 RGB images                                                     |
| Training Hardware  | NVIDIA T4 GPU (Google Colab/SageMaker)                                 |
| Metrics Tracked    | Accuracy, Precision/Recall, F1-score, ROC-AUC                          |

## Installation

```bash
# Clone repository
git clone https://github.com/marfg97/dragon-fruit-classification.git
cd dragon-fruit-classification
```

### Configure your AWS credentials
``bash
aws configure

AWS Access Key ID: [YOUR_ACCESS_KEY]
AWS Secret Access Key: [YOUR_SECRET_KEY]
Default region name: [e.g., us-west-2]
```