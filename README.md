# Brain Tumor Detection: 4-Class MRI Classification

## Problem Statement
Manual diagnosis of brain tumors from Magnetic Resonance Imaging (MRI) is a time-intensive, computationally complex, and observer-dependent task. Subtle variations in texture, structural boundaries, and pixel intensity across different tumor categories (Glioma, Meningioma, Pituitary) often lead to diagnostic delays and a high risk of inter-observer misclassification.

## The Problem This Project Solves
This system automates and standardizes brain MRI diagnostics, providing radiologists with an objective, fast, and consistent decision-support tool. By accurately categorizing scans into healthy tissue and distinct pathological classes, it reduces clinical workload, minimizes false-negative diagnoses, and aids in rapid early-stage intervention.

---

## Project Objective
The primary objective of this project is to build an end-to-end, computer vision framework that accurately classifies brain MRI scans into four distinct diagnostic categories:
1. **Glioma**
2. **Meningioma**
3. **No Tumor (Healthy Brain Tissue)**
4. **Pituitary Tumor**

The project provides a comprehensive benchmark comparing classical machine learning pipelines (using multi-domain handcrafted features) against deep convolutional transfer learning architectures.

---

## Dataset Information
* **Total Sample Size:** 7,200 brain MRI slices.
* **Class Distribution:** Perfectly balanced across 4 classes (1,800 images per class: Glioma, Meningioma, No Tumor, Pituitary).
* **Image Dimensions & Format:** Standardized RGB/grayscale MRI scans resized to a uniform 224×224 resolution.
* **Split Strategy:** Group-aware stratified splitting (`GroupShuffleSplit` and `StratifiedGroupKFold`) based on extracted patient/subject identifiers to prevent slices from the same subject from straddling the training, validation, and test partitions.

---

## Exploratory Data Analysis (EDA)
* **Class Balance Verification:** Confirmed uniform class distribution with zero class imbalance artifacts.
* **Pixel Intensity & Dimension Profiling:** Evaluated aspect ratios, channel formats, and mean pixel intensity histograms to assess scan contrast variance.
* **Visual Data Inspection:** Rendered representative multi-slice grids for each pathological category to inspect structural boundaries and tissue morphology.
* **Perceptual De-duplication:** Deployed Perceptual Hashing (`imagehash.phash`) with Hamming distance thresholding ($\le 5$) to identify and eliminate visually identical and repeated slice copies, eliminating cross-split contamination.

---

## Image Preprocessing & Feature Engineering
* **Skull Stripping:** Implemented Otsu automated thresholding combined with largest-contour extraction to remove non-brain tissues, skull artifacts, and background noise.
* **Contrast Enhancement (CLAHE):** Applied Contrast Limited Adaptive Histogram Equalization to normalize illumination and improve tumor boundary visibility in low-contrast regions.
* **Dimensionality Reduction:** Built a leak-free scikit-learn pipeline using `StandardScaler` followed by `Principal Component Analysis (PCA)` retaining 95% of the total cumulative variance.

---

## Models Trained
* **CNN**
* **VGG16**
* **ResNet50**
* **EfficientNet-B0**

---

## Evaluation Metrics
All models are evaluated on an unseen, test partition using comprehensive multi-class diagnostic metrics:
* **Overall Accuracy**
* **Macro Precision**
* **Macro Recall (Sensitivity)**
* **Macro F1-Score**
* **Confusion Matrix Analysis** (Class-specific True/False diagnostic distributions)
* **Multi-Class One-vs-Rest (OvR) ROC-AUC Curves** (Evaluating class separability across dynamic decision thresholds)
