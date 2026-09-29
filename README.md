# ♻️ Deep Learning-Based Waste Classification System
This project investigates automated household-waste image classification using four distinct deep-learning architectures under a controlled experimental design. The practical motivation is that visual recognition can support waste-sorting workflows, recycling assistance, smart-bin interfaces, and educational disposal tools. using **Custom CNN, ResNet50, EfficientNet, and Vision Transformer (ViT)** architectures.

The project investigates how different deep learning approaches perform on a multi-class waste recognition problem while maintaining a consistent dataset, preprocessing pipeline, train/validation/test split, and evaluation methodology.

---

## 📌 Project Overview

Effective waste classification is an important component of modern waste management and recycling systems. Manual waste identification can be time-consuming, inconsistent, and difficult to scale, particularly when large quantities of visually diverse waste must be processed.

This project explores the use of **deep learning and computer vision** to automatically classify waste images into ten categories.

The system is developed using **PyTorch** and evaluates four different deep learning architectures:

1. Custom Convolutional Neural Network (CNN)
2. ResNet50
3. EfficientNet
4. Vision Transformer (ViT)

Each model is developed independently while using the same shared data pipeline and dataset partitions to support a fair comparison.

---

## 🎯 Project Objectives

The main objectives of this project are to:

- Develop an automated multi-class waste image classification system.
- Perform systematic dataset auditing and exploratory data analysis.
- Build a reusable PyTorch preprocessing and data-loading pipeline.
- Compare CNN, residual, efficiency-oriented, and transformer-based architectures.
- Investigate the effectiveness of transfer learning.
- Evaluate models using multiple classification metrics rather than accuracy alone.
- Analyze class-level performance and misclassification behaviour.
- Measure computational characteristics such as inference time and throughput.
- Support prediction on previously unseen waste images.
- Establish a foundation for future real-time waste recognition.

---

## 🗂️ Waste Categories

The dataset contains **10 waste categories**:

| ID | Class |
|---:|---|
| 0 | Battery |
| 1 | Biological |
| 2 | Cardboard |
| 3 | Clothes |
| 4 | Glass |
| 5 | Metal |
| 6 | Paper |
| 7 | Plastic |
| 8 | Shoes |
| 9 | Trash |

---

## 📊 Dataset

The final dataset contains:

**19,762 supported images**

The class distribution identified during dataset auditing was:

| Class | Images | Percentage |
|---|---:|---:|
| Battery | 944 | 4.78% |
| Biological | 997 | 5.05% |
| Cardboard | 1,825 | 9.23% |
| Clothes | 5,327 | 26.96% |
| Glass | 3,061 | 15.49% |
| Metal | 1,020 | 5.16% |
| Paper | 1,680 | 8.50% |
| Plastic | 1,984 | 10.04% |
| Shoes | 1,977 | 10.00% |
| Trash | 947 | 4.79% |

The largest class is **Clothes**, containing 5,327 images, while the smallest class is **Battery**, containing 944 images.

This produces an approximate class imbalance ratio of:

```text
5.64 : 1
```

Because of this imbalance, model evaluation includes **macro-averaged metrics** in addition to overall accuracy.

---

## 🔍 Dataset Audit

Before model development, the dataset was systematically inspected.

The audit included:

- File-extension analysis
- Unsupported-file detection
- Corrupted/unreadable image detection
- Image colour-mode analysis
- Native image dimension analysis
- Aspect-ratio analysis
- Exact duplicate detection
- Cross-class duplicate checking
- Class-distribution analysis

### Key Findings

```text
Total supported images:     19,762
Corrupted images:           0
Unsupported files:          0
Exact duplicate groups:     0
Cross-class exact overlap:  0
```

Most images are stored in RGB format, although the dataset also contains RGBA, palette-based, CMYK, and grayscale images.

Native image dimensions and aspect ratios vary considerably. These differences are handled by the shared preprocessing pipeline before images are supplied to the models.

---

## ✂️ Dataset Split

A **stratified train/validation/test split** was created to preserve class distributions across the three partitions.

| Partition | Images |
|---|---:|
| Training | 13,833 |
| Validation | 2,964 |
| Test | 2,965 |
| **Total** | **19,762** |

The split manifests are stored in:

```text
data/manifests/
├── train.csv
├── val.csv
├── test.csv
└── split_summary.csv
```

Data leakage checks confirmed:

```text
Train ↔ Validation overlap: 0
Train ↔ Test overlap:       0
Validation ↔ Test overlap:  0
```

---

## ⚙️ Data Preprocessing

A shared PyTorch preprocessing pipeline is used to maintain consistency across model experiments.

The pipeline handles the variability found during dataset auditing by standardizing images before they are passed to the models.

Typical processing includes:

```text
Input Image
     │
     ▼
RGB Conversion
     │
     ▼
Resize / Spatial Transformation
     │
     ▼
224 × 224 Input
     │
     ▼
Tensor Conversion
     │
     ▼
Normalization
     │
     ▼
Model Input
```

For transfer-learning models, preprocessing is designed to remain compatible with the corresponding pretrained representations.

Training data additionally uses augmentation to introduce controlled variation and improve generalization, while validation and test data use deterministic evaluation transformations.

A standard batch has the shape:

```text
torch.Size([32, 3, 224, 224])
```

representing:

```text
Batch × Channels × Height × Width
```

---

# 🧠 Model Architectures

Four architectures are investigated.

## 1. Custom CNN

The Custom CNN provides a task-specific convolutional baseline trained for the waste classification problem.

It allows the project to compare representations learned directly from the target dataset against pretrained architectures.

> **Status:** Developed by the assigned team member.

---

## 2. ResNet50

ResNet50 uses residual connections to support optimization of deep convolutional networks.

The project uses an **ImageNet-pretrained ResNet50** and adapts the final classifier for the ten waste categories.

### Modified Classification Head

```text
ResNet50 Backbone
       │
       ▼
Global Average Pooling
       │
       ▼
2048-D Feature Vector
       │
       ▼
Linear(2048 → 10)
       │
       ▼
10 Waste Classes
```

The modified network contains:

```text
Total parameters:     23,528,522
Trainable Stage 1:        20,490
Frozen Stage 1:       23,508,032
```

---

## 3. EfficientNet

EfficientNet is included as an efficiency-oriented convolutional architecture.

It provides an opportunity to investigate the trade-off between predictive performance and computational requirements.

> **Status:** Developed by the assigned team member.

---

## 4. Vision Transformer (ViT)

Vision Transformer represents images as sequences of patches and applies transformer-based self-attention rather than relying entirely on conventional convolutional feature extraction.

Its inclusion provides architectural diversity in the model comparison.

> **Status:** Developed by the assigned team member.

---

# 🔬 ResNet50 Experimental Strategy

The ResNet50 experiment was performed in two stages.

## Stage 1 — Frozen Backbone

An ImageNet-pretrained ResNet50 was first used as a fixed feature extractor.

```text
Input
  │
  ▼
Conv1      ───── Frozen
  │
  ▼
Layer1     ───── Frozen
  │
  ▼
Layer2     ───── Frozen
  │
  ▼
Layer3     ───── Frozen
  │
  ▼
Layer4     ───── Frozen
  │
  ▼
Global Average Pooling
  │
  ▼
FC Layer   ───── Trainable
  │
  ▼
10 Classes
```

Only the newly introduced classification layer was optimized.

### Stage 1 Result

The best checkpoint occurred at **epoch 9**:

```text
Training Loss:       0.1391
Training Accuracy:   95.55%

Validation Loss:     0.1697
Validation Accuracy: 94.97%
```

Validation metrics:

| Metric | Score |
|---|---:|
| Accuracy | 94.97% |
| Macro Precision | 93.72% |
| Macro Recall | 94.32% |
| Macro F1 | 93.94% |
| Weighted F1 | 94.96% |

---

## Stage 2 — Partial Fine-Tuning

After the frozen-backbone stage, the deeper ResNet50 features were partially adapted to the waste dataset.

```text
Input
  │
  ▼
Conv1      ───── Frozen
  │
  ▼
Layer1     ───── Frozen
  │
  ▼
Layer2     ───── Frozen
  │
  ▼
Layer3     ───── Frozen
  │
  ▼
Layer4     ───── Trainable
  │
  ▼
Global Average Pooling
  │
  ▼
FC Layer   ───── Trainable
  │
  ▼
10 Classes
```

A smaller learning rate was used during this stage to update the pretrained representation more conservatively.

### Best Fine-Tuned Validation Result

```text
Best epoch:          3
Validation loss:     0.1280
Validation accuracy: 96.59%
```

| Metric | Score |
|---|---:|
| Accuracy | 96.59% |
| Macro Precision | 95.89% |
| Macro Recall | 95.84% |
| Macro F1 | 95.84% |
| Weighted F1 | 96.60% |
| Macro ROC-AUC | 0.9986 |
| Weighted ROC-AUC | 0.9987 |

Partial fine-tuning therefore improved the validation performance compared with the frozen-backbone experiment.

---

# 🏆 Final ResNet50 Test Results

After model selection was completed using the validation partition, the selected fine-tuned checkpoint was evaluated on the independent **2,965-image test set**.

| Metric | Result |
|---|---:|
| Accuracy | **95.24%** |
| Macro Precision | **94.36%** |
| Macro Recall | **94.36%** |
| Macro F1 | **94.34%** |
| Weighted F1 | **95.23%** |
| Macro ROC-AUC | **0.9977** |
| Weighted ROC-AUC | **0.9981** |

The test partition was kept separate from model selection to provide a more reliable estimate of generalization to unseen samples.

---

## ⏱️ ResNet50 Computational Performance

The final model was also evaluated in terms of computational characteristics.

```text
Total parameters:       23,528,522
Parameters:             23.53M
Checkpoint size:        204.42 MB

Test images:            2,965
Total inference time:   22.57 s
Average inference:      7.61 ms/image
Throughput:             131.39 images/s
```

These measurements were obtained in the project's experimental environment and may vary depending on hardware, batch size, software configuration, and inference methodology.

---

# 📈 Evaluation Metrics

Models are evaluated using multiple complementary metrics:

- Accuracy
- Precision
- Recall
- Macro Precision
- Macro Recall
- Macro F1-score
- Weighted F1-score
- Confusion Matrix
- Per-class classification metrics
- ROC curves
- Macro ROC-AUC
- Weighted ROC-AUC
- Parameter count
- Inference latency
- Throughput

Macro-averaged metrics are particularly important because the dataset is imbalanced and they give equal importance to each class.

---

# 🖼️ Single-Image Prediction

The final ResNet50 model supports classification of individual unseen images.

The inference pipeline follows:

```text
New Waste Image
       │
       ▼
Load Image
       │
       ▼
Convert to RGB
       │
       ▼
Evaluation Transform
       │
       ▼
[1, 3, 224, 224]
       │
       ▼
Fine-Tuned ResNet50
       │
       ▼
Softmax Probabilities
       │
       ▼
Predicted Class + Confidence
```

This provides the foundation for future interactive applications.

---

# 📷 Future Real-Time Classification

A planned extension is **real-time webcam-based waste recognition**.

The proposed pipeline is:

```text
Camera Frame
     │
     ▼
Image Preprocessing
     │
     ▼
ResNet50 Inference
     │
     ▼
Predicted Waste Category
     │
     ▼
Confidence Score
     │
     ▼
Real-Time Display
```

Future development could also investigate:

- Unknown-class rejection
- Confidence thresholds
- Object detection
- Multiple-object recognition
- Semantic/instance segmentation
- Model quantization
- Model pruning
- Mobile or edge deployment
- External dataset validation

---

# 📁 Project Structure

```text
waste-classification/
│
├── data/
│   ├── raw/
│   │   └── garbage-dataset/
│   │
│   └── manifests/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── split_summary.csv
│
├── notebooks/
│   └── ...
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── dataloader.py
│   │   └── transforms.py
│   │
│   ├── models/
│   │   └── ...
│   │
│   └── utils/
│       └── seed.py
│
├── results/
│   ├── checkpoints/
│   │   ├── resnet50_frozen_best.pth
│   │   ├── resnet50_layer4_finetuned_best.pth
│   │   └── resnet50_weights_only.pth
│   │
│   ├── tables/
│   │   ├── resnet50_frozen_validation_report.csv
│   │   ├── resnet50_finetuned_validation_report.csv
│   │   ├── resnet50_finetuned_validation_summary.csv
│   │   ├── resnet50_final_test_classification_report.csv
│   │   └── resnet50_final_test_summary.csv
│   │
│   └── ...
│
├── experiments/
│   └── ...
│
├── requirements.txt
├── .gitignore
└── README.md
```

> The exact contents of model-specific directories may evolve as the remaining architectures are completed.

---

# 💻 Technology Stack

The project uses:

- **Python**
- **PyTorch**
- **Torchvision**
- **Scikit-learn**
- **NumPy**
- **Pandas**
- **Pillow**
- **Matplotlib**
- **Jupyter**
- **Visual Studio Code**
- **Git**
- **GitHub**
- **CUDA / NVIDIA GPU acceleration**

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone <repository-url>
cd waste-classification
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU acceleration, ensure that the installed PyTorch build is compatible with the system's supported CUDA environment.

---

## 4. Prepare the Dataset

The expected local dataset structure is:

```text
data/raw/garbage-dataset/
├── battery/
├── biological/
├── cardboard/
├── clothes/
├── glass/
├── metal/
├── paper/
├── plastic/
├── shoes/
└── trash/
```

The dataset itself should not necessarily be committed to Git if it is large. Follow the project's dataset setup instructions and licensing requirements.

---

## 5. Verify the Data Pipeline

The shared DataLoader can be used to verify the dataset and manifests.

Example:

```python
from src.data.dataloader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    dataset_root="data/raw/garbage-dataset",
    manifest_dir="data/manifests",
    batch_size=32,
    num_workers=0,
    pin_memory=True
)
```

Expected dataset sizes:

```text
Train:      13,833
Validation: 2,964
Test:       2,965
```

---

# 💾 ResNet50 Checkpoints

The ResNet50 experiments produce different checkpoints for different purposes.

```text
results/checkpoints/
│
├── resnet50_frozen_best.pth
│      └── Best Stage 1 frozen-backbone checkpoint
│
├── resnet50_layer4_finetuned_best.pth
│      └── Best partially fine-tuned checkpoint
│
└── resnet50_weights_only.pth
       └── Weights-only model representation
```

For final ResNet50 evaluation, the selected model is:

```text
resnet50_layer4_finetuned_best.pth
```

---

# 🔁 Reproducibility

To improve reproducibility:

- Dataset partitions are stored as CSV manifests.
- All models use the same train/validation/test partitions.
- Shared preprocessing functions are used where appropriate.
- Random seeds are controlled through project utilities.
- Model selection is performed using validation data.
- Final evaluation is performed separately on the test partition.
- Checkpoints and result summaries are stored under `results/`.

This separation is important to reduce accidental data leakage and maintain a fair comparison between architectures.

---

# ⚠️ Limitations

Although the current results are promising, several limitations should be considered.

The dataset is moderately imbalanced, and the ten categories do not all represent waste at exactly the same semantic level. Some categories represent materials, while others represent specific object types.

The current system also performs **single-label, closed-set image classification**. It assumes that an input image belongs to one of the ten known categories.

Real-world waste environments may contain:

- Multiple objects in one image
- Mixed-material objects
- Occlusion
- Cluttered backgrounds
- Poor or changing lighting
- Damaged or contaminated objects
- Unseen waste categories
- Different camera conditions

Therefore, strong test-set performance should not automatically be interpreted as equivalent performance in unrestricted real-world deployment.

---

# 👥 Team Development

The project is divided so that each team member can develop their architecture independently while relying on the same shared experimental foundation.

| Component | Responsibility |
|---|---|
| Shared dataset preparation | masterGB(Gihan) |
| Dataset auditing / EDA | masterGB(Gihan) |
| Train/validation/test split | masterGB(Gihan) |
| Shared preprocessing pipeline | masterGB(Gihan) |
| Custom CNN | Team Member |
| ResNet50 | masterGB(Gihan |
| EfficientNet | Team Member |
| Vision Transformer | Team Member |
| Final model comparison | Team |
| Report and presentation | Team |

This structure minimizes dependencies between individual model-development tasks and allows members to work concurrently.

---

# 🔀 Git Workflow

Individual architectures should be developed on separate branches.

Example:

```text
main
├── custom-cnn
├── resnet50
├── efficientnet
└── vit
```

A typical workflow is:

```bash
git checkout main
git pull origin main

git checkout <your-branch>

git add .
git commit -m "descriptive commit message"
git push origin <your-branch>
```

Model-specific work should be reviewed and tested before integration into the shared `main` branch.

---

# 🧪 Experimental Principles

The project follows several important experimental principles:

1. **Consistent data partitions** across all models.
2. **No train/validation/test overlap.**
3. **Validation data for model selection.**
4. **Test data reserved for final evaluation.**
5. **Multiple evaluation metrics**, not accuracy alone.
6. **Macro metrics** to account for class imbalance.
7. **Checkpointing based on validation performance.**
8. **Reproducible preprocessing and dataset manifests.**
9. **Computational evaluation alongside predictive performance.**
10. **Critical analysis of limitations and failure cases.**

---

# 📚 Model References

The main architectures and techniques used in this project are based on established deep-learning research:

- K. He, X. Zhang, S. Ren, and J. Sun, **“Deep Residual Learning for Image Recognition,”** CVPR, 2016.
- M. Tan and Q. V. Le, **“EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,”** ICML, 2019.
- A. Dosovitskiy et al., **“An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale,”** ICLR, 2021.
- J. Deng et al., **“ImageNet: A Large-Scale Hierarchical Image Database,”** CVPR, 2009.
- A. Paszke et al., **“PyTorch: An Imperative Style, High-Performance Deep Learning Library,”** NeurIPS, 2019.

---

# 📝 Current Project Status

| Component | Status |
|---|:---:|
| Dataset audit | ✅ |
| Corruption checking | ✅ |
| Duplicate checking | ✅ |
| Stratified dataset split | ✅ |
| Leakage verification | ✅ |
| Shared preprocessing | ✅ |
| Shared DataLoader | ✅ |
| ResNet50 frozen training | ✅ |
| ResNet50 partial fine-tuning | ✅ |
| ResNet50 validation evaluation | ✅ |
| ResNet50 final test evaluation | ✅ |
| ResNet50 single-image prediction | ✅ |
| Custom CNN | ✅ |
| EfficientNet | ✅ |
| Vision Transformer | ✅ |
| Four-model comparison | ✅ |
| Real-time camera prototype | 🔮 Future Work |

---

# 📌 Final Note

This repository is intended to provide a structured and reproducible experimental framework for comparing multiple deep-learning approaches to waste classification.

The current ResNet50 experiment demonstrates that transfer learning can provide strong performance on this dataset, with the final partially fine-tuned model achieving:

```text
Test Accuracy:       95.24%
Macro F1:            94.34%
Weighted F1:         95.23%
Macro ROC-AUC:       0.9977
Weighted ROC-AUC:    0.9981
```

The final project comparison will incorporate the results of all four architectures and analyze their predictive performance, computational characteristics, and practical trade-offs.
