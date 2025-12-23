# Multimodal Facial Aesthetics Assessment Framework with Deep Learning and Geometric Modeling

## Abstract
This repository presents a comprehensive, research-oriented framework for automated facial aesthetics assessment using a multimodal learning strategy. The system integrates deep convolutional neural networks, geometric facial landmark analysis, handcrafted morphological features, and an ensemble-based decision mechanism to predict continuous aesthetic scores. Built upon the SCUT-FBP5500 dataset, the framework combines appearance-based learning with interpretable facial structure modeling, enabling both high predictive performance and analytical transparency. The implementation reflects an end-to-end research pipeline, from data ingestion and preprocessing to feature engineering, model training, ensemble fusion, and deployment-ready artifact generation.

---

## Problem Formulation
Facial attractiveness assessment is a complex perceptual task influenced by multiple factors, including facial symmetry, proportions, texture, and learned aesthetic priors. Traditional single-model approaches often fail to capture this multifaceted nature. This work formulates facial aesthetics prediction as a continuous regression problem and addresses it through a multimodal architecture that fuses learned visual representations with explicit geometric and morphological descriptors derived from facial landmarks.

---

## Dataset and Preprocessing
The framework is developed using the SCUT-FBP5500 dataset, consisting of facial images annotated with continuous attractiveness scores. Images are loaded from disk using metadata stored in CSV files, with preprocessing steps including:
- RGB conversion and resizing to standardized input dimensions
- Pixel-level normalization
- Train–validation splitting to ensure unbiased evaluation

Sample visualizations are included to validate dataset integrity and preprocessing correctness.

---

## Model A: Appearance-Based Deep Regression
The first predictive branch employs a pretrained EfficientNet-V2-S backbone initialized with ImageNet weights. The classification head is replaced with a regression layer to output a single continuous attractiveness score. This model learns high-level visual and texture cues directly from facial images and is optimized using mean squared error loss with the Adam optimizer. The trained model is saved and reused for inference and ensemble integration.

---

## Facial Landmark Extraction and Geometric Feature Engineering
To introduce interpretability and structural awareness, MediaPipe FaceMesh is used to extract 468 dense facial landmarks per image. From these landmarks, a set of engineered geometric and morphological features is computed, including:
- Eye aspect ratios and bilateral symmetry
- Mouth width and lip thickness
- Nose width and proportionality
- Jawline curvature statistics
- Cheek prominence relative to facial midline
- Skin sharpness via Laplacian variance in localized facial regions
All scale-dependent features are normalized relative to estimated face width to ensure invariance to image scale.

---

## Feature Normalization and Caching
Extracted feature vectors are standardized using dataset-wide mean and standard deviation statistics. Both raw and normalized features are cached to disk to support efficient experimentation, reproducibility, and modular reuse across models.

---

## Model B: Interpretable Subscore-Based Regression
A second predictive branch operates purely on engineered facial features. This model uses a multilayer perceptron to generate multiple latent sub-scores corresponding to interpretable facial attributes (e.g., eyes, jawline, nose, lips, cheeks, skin). These sub-scores are constrained to a calibrated range and combined via learnable softmax-normalized weights to produce an overall attractiveness prediction. This design explicitly models the relative contribution of different facial components.

---

## CNN-Enhanced Feature Fusion
To further enrich Model B, a lightweight convolutional backbone based on MobileNetV3 is introduced to extract compact visual embeddings. These embeddings are concatenated with normalized geometric features and passed through an extended regression head, allowing the model to jointly reason over learned texture cues and explicit facial structure.

---

## Final Ensemble Architecture
The final system integrates:
- Model A (appearance-based deep regressor)
- Model B (feature-based subscore regressor with CNN enhancement)

A minimal trainable combiner module learns to fuse predictions from both branches using a learnable weighting parameter and a shallow correction head. During ensemble training, all base models are frozen, and only the combiner parameters are optimized, ensuring stability and preventing catastrophic forgetting.

---

## Training and Evaluation
Training procedures include:
- Mean squared error optimization
- Validation-based monitoring using MAE and RMSE
- Lightweight regularization to prevent sub-score saturation
- Visualization of predicted versus ground-truth scores
- Statistical analysis of sub-score distributions

This multi-level evaluation provides both quantitative performance assessment and qualitative insight into model behavior.

---

## Inference and Deployment
The repository includes utilities for single-image inference, producing:
- Overall attractiveness prediction
- Interpretable sub-scores
- Learned sub-score combination weights
- Optional landmark visualizations

All trained models, normalization parameters, and ensemble states are exported into a deployment directory, enabling seamless downstream integration.

---

## Contributions
- A complete multimodal facial aesthetics assessment pipeline
- Integration of deep visual models with explicit geometric facial analysis
- Interpretable sub-score modeling with learnable aggregation
- Ensemble-based fusion of heterogeneous predictors
- Reproducible, deployment-ready research implementation

---

## Technologies
Python, PyTorch, Torchvision, MediaPipe, OpenCV, NumPy, Pandas, Scikit-learn, Matplotlib, TQDM, Hugging Face Hub

---

## Deployed App for DEMO: https://huggingface.co/spaces/Shizu75/faced

## Author
This repository reflects a research-driven implementation of multimodal perception modeling, emphasizing methodological rigor, interpretability, and reproducibility.
