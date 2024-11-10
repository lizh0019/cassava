# Cassava Leaf Disease Recognition System

## Overview
This project provides a machine learning pipeline for classifying cassava leaf diseases using image segmentation, feature extraction, and an ensemble deep learning classifier. The approach includes both traditional and deep learning-based segmentation methods, a Progressive Learning Algorithm (PLA) for feature extraction, and an ensemble model for classification. It achieved best performance in Kaggle 2020 competition Cassava Leaf Disease Classification to identify the type of disease present on a Cassava Leaf image.

## Project Structure
cassava code/
├── ensemble-classsification
│   ├── input
│   │   ├── cassava-leaf-disease-classification
│   │   ├── image-fmix
│   │   ├── pytorch-image-models
│   │   └── timm-pytorch-image-models -> pytorch-image-models/
│   ├── results
│   │   ├── ag_model
│   │   ├── best_model
│   │   └── __results___files
│   └── src
│       ├── checkpoint
│       ├── dask-worker-space
│       └── __pycache__
├── feature-extraction-PLA
│   ├── heads
│   │   └── __pycache__
│   ├── models
│   ├── nets
│   │   └── __pycache__
│   ├── __pycache__
│   └── query
└── segmentation
    ├── GMM
    │   ├── gbvs
    │   ├── leaf toy examples
    │   └── vlfeat-0.9.9
    └── unet
        ├── docs
        ├── notebooks
        ├── scripts
        ├── src
        └── tests

#Main components:

## 1. Segmentation
The segmentation pipeline uses two methods:
- **Gaussian Mixture Model (GMM):** Segments cassava leaves by clustering pixel features.
- **U-Net Model:** Self-supervised training with pseudo-masks from GMM to refine segmentation results.

## 2. Feature Extraction
The **Progressive Learning Algorithm (PLA)** generates feature embeddings optimized by combining cross-entropy and triplet loss functions. These embeddings serve as robust inputs for classification.

## 3. Deep Learning Ensemble Classification
An ensemble of deep learning models (e.g., ResNet-50, EfficientNet) aggregates predictions to improve classification accuracy on cassava leaf disease categories.

## Setup Instructions

### Requirements
- **Python 3.7+**
- **Required Libraries**: `torch`, `torchvision`, `scikit-learn`, `numpy`, `pandas`


#Repository:

cassava-leaf-disease-recognition available in repository <git@github.com:lizh0019/cassava.git>

#Prepare Data:

Training and testing images available in <https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data>
Simply login Kaggle and download the dataset.

##Dataset structure:

test_images
test_tfrecords
train_images
train_tfrecords
sample_submission.csv
train.csv
label_num_to_disease_map.json:
"root":{
"0":string"Cassava Bacterial Blight (CBB)"
"1":string"Cassava Brown Streak Disease (CBSD)"
"2":string"Cassava Green Mottle (CGM)"
"3":string"Cassava Mosaic Disease (CMD)"
"4":string"Healthy"
}

#Implementation steps

##Segmentation:

GMM segmentation:
segmentation/GMM/leaf_segmentation.py
Train the U-Net model:
unet/src/train_unet.py

##Feature Extraction:

Extract features using PLA:
feature-extraction-PLA/extract_features.py
Train the Ensemble Classifier:

Train or use the pre-trained ensemble model:
ensemble-classification/train_ensemble.py

##Classification:
ensemble-classification/infer.py

##Evaluation
Result models are saved in ensemble-classification/results. 
Analyze model performance with:
ensemble-classification/evaluate_results.py

#Acknowledgments

This project uses resources from the PyTorch Image Models (TIMM), FMix for augmentations, and traditional image processing libraries (VLFeat).

