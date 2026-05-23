# Midterm Project — Image Classification with Machine Learning Pipeline

**Course:** Machine Vision (RE604)  
**Name:** Dhini Ari Minarti  
**Student ID:** 4222311022  

---

## Project Description

This project builds an image classification pipeline for handwritten letters A–Z using the **EMNIST Letters** dataset. The pipeline combines **HOG (Histogram of Oriented Gradients)** feature extraction with a **Support Vector Machine (SVM)** classifier optimized via **Grid Search**.

```
Dataset CSV → Balanced Sampling → HOG Extraction → StandardScaler → SVM (Grid Search) → Evaluation
```

---

## Repository Structure

```
├── midterm_emnist.ipynb          # Main notebook (run this)
├── emnist-letters-train.csv      # Dataset (download from Kaggle)
├── README.md                     # This documentation
└── output/                       # Generated visualizations
    ├── 01_sample_images.png
    ├── 02_hog_visualization.png
    ├── 03_gridsearch_heatmap.png
    ├── 04_confusion_matrix.png
    └── 05_performance_summary.png
```

---

## Dataset

- **Name:** EMNIST Letters Dataset
- **Source:** [Kaggle — EMNIST](https://www.kaggle.com/datasets/crawford/emnist/data)
- **File used:** `emnist-letters-train.csv`
- **Total data:** 88,800 rows, 785 columns
- **Format:** First column = class label (1–26), columns 2–785 = pixel values (28×28 = 784 pixels)
- **Classes:** 26 letters (A–Z, uppercase and lowercase merged into one class per letter)

---

## Requirements

Install all dependencies using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image jupyter
```

| Library | Minimum Version | Purpose |
|---|---|---|
| `numpy` | 1.21+ | Array operations |
| `pandas` | 1.3+ | Load & manipulate CSV |
| `matplotlib` | 3.4+ | Visualization |
| `seaborn` | 0.11+ | Heatmaps & plots |
| `scikit-learn` | 1.0+ | SVM, Grid Search, evaluation metrics |
| `scikit-image` | 0.18+ | HOG feature extraction |

---

## How to Run

1. **Clone / download** this repository
2. **Download the dataset** from Kaggle and place `emnist-letters-train.csv` in the same folder as the notebook
3. **Open the notebook:**
   ```bash
   jupyter notebook midterm_emnist.ipynb
   ```
4. **Update the dataset path** in Cell 1 if needed:
   ```python
   TRAIN_CSV = 'emnist-letters-train.csv'  # adjust path if necessary
   ```
5. **Run all cells:** `Kernel → Restart & Run All`

> Grid Search may take approximately 3–5 minutes depending on your machine's specifications.

---

## Pipeline Explanation

### 1. Dataset Preparation
- Load the EMNIST Letters CSV (88,800 samples)
- Sample **100 instances per class** randomly → total **2,600 samples** (26 classes × 100)
- **Shuffle** the entire dataset before processing
- **Transpose** each image because EMNIST stores pixel values in a flipped orientation
- Split into **80% Training (2,080)** / **20% Testing (520)** using stratified splitting

### 2. HOG Feature Extraction

HOG captures local shape and edge information from an image through three steps:
1. Compute pixel gradients (direction & magnitude)
2. Build orientation histograms per small cell
3. Normalize histograms per block to handle lighting variations

**Tuned parameters (changed from defaults):**

| Parameter | Default | **Chosen Value** | Reason |
|---|---|---|---|
| `orientations` | 9 | **12** | More orientation bins → richer shape description |
| `pixels_per_cell` | (8, 8) | **(4, 4)** | Smaller cells → higher spatial resolution |
| `cells_per_block` | (3, 3) | **(2, 2)** | Better suited for 28×28 image size |

**Result:** **1,728 features** per image (vs. 324 features with default parameters)

### 3. SVM + Grid Search

Grid Search automatically finds the best SVM parameter combination using 5-Fold Cross-Validation:

| Parameter | Values Tested | Description |
|---|---|---|
| `kernel` | `rbf`, `linear` | Kernel function |
| `C` | 0.1, 1, 10 | Regularization — higher value = stricter fit to training data |
| `gamma` | `scale`, `auto` | Influence range of a single training sample (RBF kernel) |

**Best parameters found:** `kernel=rbf`, `C=10`, `gamma=scale`

### 4. Evaluation

Evaluation is performed using two approaches:
- **5-Fold Stratified Cross-Validation** — estimates model generalization across the full dataset
- **80/20 Train/Test Split** — final evaluation on unseen data

> **Note on LOOCV:** Leave-One-Out CV on 2,600 samples requires 2,600 separate SVM training runs (>1 hour). 5-Fold Stratified CV is a widely accepted academic approximation of LOOCV that is far more computationally efficient while still providing reliable generalization estimates.

---

## Results

### Performance Summary

| Metric | Train Set | 5-Fold CV | Test Set |
|---|---|---|---|
| **Accuracy** | 1.0000 | 0.8200 | **0.8154** |
| **Precision** | 1.0000 | 0.8232 | **0.8249** |
| **Recall** | 1.0000 | 0.8200 | **0.8154** |
| **F1-Score** | 1.0000 | 0.8201 | **0.8144** |

### Per-Class Highlights

| Letter | F1-Score | Note |
|---|---|---|
| O, S | 1.00 | Most distinctive shapes, easiest to recognize |
| Z, P, F | ≥ 0.90 | Very good performance |
| Q, G, L | ≤ 0.64 | Frequently confused with visually similar letters |

---

## Output Visualizations

| File | Description |
|---|---|
| `01_sample_images.png` | One sample image per class (A–Z) |
| `02_hog_visualization.png` | Original image vs. HOG feature map comparison |
| `03_gridsearch_heatmap.png` | CV Accuracy heatmap from Grid Search |
| `04_confusion_matrix.png` | 26×26 confusion matrix on the test set |
| `05_performance_summary.png` | Bar chart comparing Train / CV / Test metrics |

---

## Technical Notes

- `RANDOM_SEED = 42` is used across all random processes to ensure **reproducibility**
- EMNIST images must be **transposed** `(0, 2, 1)` because pixels are stored in a flipped format in the original CSV
- **StandardScaler** is applied after HOG extraction to normalize feature distributions before feeding into SVM
- A training accuracy of 100% is expected behavior for SVM with a high C value — always refer to CV and Test scores for true generalization performance

---

## References

- Cohen, G., et al. (2017). EMNIST: Extending MNIST to handwritten letters. *IJCNN*.
- Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. *CVPR*.
- Dataset: https://www.kaggle.com/datasets/crawford/emnist/data
- scikit-image HOG documentation: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog
