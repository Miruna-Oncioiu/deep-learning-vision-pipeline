# Computer Vision Classification Pipeline (PyTorch)

End-to-end deep learning project developed for the **Artificial Intelligence** course at Faculty of Automatic Control and Computer Science, UPB.

## Project Overview
The repository implements, trains, and evaluates deep neural architectures on two challenging, class-imbalanced image classification datasets:
1. **Facial Emotion Recognition ("You're on candid camera"):** 7-class facial emotion classification (~15.3k images) under unconstrained lighting and occlusions.
2. **Skin Lesion Diagnosis ("Vai de pielea mea"):** 7-class dermoscopic skin mole and lesion classification (~10k images) with severe class imbalance.

## Key Architectures & Techniques
- **Architectures Evaluated:**
  * Baseline Multi-Layer Perceptron (MLP) with dropout regularization.
  * Custom Convolutional Neural Network (CNN) featuring Batch Normalization, Max Pooling, and Adaptive Average Pooling.
  * Transfer Learning & Fine-Tuning: Pre-trained ResNet-18 and MobileNet-v2 with custom classification heads and learning rate warmup schedules.
- **Class Imbalance Mitigation:** WeightedRandomSampler, class-weighted Cross-Entropy, and Focal Loss to penalize minority-class misclassifications.
- **Data Augmentation:** Geometric transforms (flips, random rotations, crops) and color space adjustments tailored per task domain via `torchvision.transforms` / `Albumentations`.
- **Experiment Tracking & Ablations:** Systematic ablation studies logged via TensorBoard (monitoring loss, top-1 accuracy, per-class F1-scores, and confusion matrices).

## Tech Stack
- **Framework:** PyTorch, Torchvision
- **Visualization & Logging:** TensorBoard, Matplotlib, Seaborn
- **Preprocessing & Data:** NumPy, Pandas, PIL, Albumentations
