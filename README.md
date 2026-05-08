# Improved Credit Score Classification Using Ensemble Learning with Bayesian Hyperparameter Optimization

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Paper:** *Improved Credit Score Classification Using Ensemble Learning with Bayesian Hyperparameter Optimization*  
> **Author:** Ajete Isaku — Faculty of Mathematics and Natural Sciences, University of Pristina  
> **Dataset:** [Kaggle Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) by Rohan Paris (2022)

---

## Overview

This repository contains the full implementation for a research paper that identifies and corrects a **critical data leakage problem** in the widely-used Kaggle Credit Score Classification dataset. The dataset contains 100,000 rows representing 12,500 unique customers × 8 monthly snapshots. Prior studies split at the row level, leaking the same customer into both train and test sets and inflating reported accuracy by 6–10 percentage points.

Our corrected pipeline achieves on **properly deduplicated data (12,500 customers)**:

| Metric | XGBoost | Random Forest | Stacking (XGB+RF) |
|--------|---------|---------------|-------------------|
| Accuracy | **74.24%** | 73.84% | 74.08% |
| Macro-F1 | **73.86%** | 73.49% | 73.75% |
| ROC-AUC | **87.56%** | 86.22% | 86.45% |
| 5-Fold CV F1 | **95.48 ± 0.26%** | 94.72 ± 0.52% | — |
| Runtime | **386s** | — | — |

**103× faster** than the original WOA-LSTM approach (39,572s).

---

## Key Findings

1. **Data Leakage Discovery** — The 100K rows = 12,500 customers × 8 months. Row-level splitting inflates accuracy by ~6–10 percentage points.
2. **Customer-Level Aggregation** — Median (numeric) and mode (categorical) per customer eliminates leakage.
3. **Feature Engineering** — 6 domain-informed financial ratios + variability features → 52 total features.
4. **SMOTEENN** — Combined oversampling + noise cleaning outperforms standalone SMOTE.
5. **Bayesian Optimization** — Optuna TPE converges in ~10 trials vs. WOA's 65 evaluations.
6. **XGBoost Dominance** — Stacking adds no meaningful improvement over well-tuned XGBoost.

---

## Project Structure

```
credit-score-classification/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── credit_score_improved.py           # Full pipeline (standalone version)
├── pipeline_v2_real_data.py           # Pipeline with customer-level aggregation
│
├── data/
│   └── train.csv                      # Kaggle dataset (download separately)
│
├── figures/                           # Generated figures for the paper
│   ├── fig1_pipeline.png              # Pipeline architecture diagram
│   ├── fig2_leakage.png               # Data leakage illustration
│   ├── fig3_distribution.png          # Class distribution before/after
│   ├── fig4_smoteenn.png              # SMOTEENN resampling effect
│   ├── fig5_comparison.png            # Multi-metric model comparison
│   ├── fig6_confusion.png             # Confusion matrices
│   ├── fig7_cv_runtime.png            # CV vs holdout + runtime
│   ├── fig8_features.png              # Feature importance
│   └── fig9_optuna.png                # Optuna convergence
│
├── paper/
│   ├── Credit_Score_Springer_LNCS.docx  # Paper (Springer LNCS format)
│   ├── paper_source.tex                 # LaTeX source (IEEE format)
│   └── Presentation_Script.docx         # Speaker notes for presentation
│
└── results/
    └── results.json                   # Saved experiment results
```

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/ajete-isaku/credit-score-classification.git
cd credit-score-classification
pip install -r requirements.txt
```

### 2. Download the Dataset

Download `train.csv` from [Kaggle](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) and place it in the `data/` directory.

### 3. Run the Pipeline

```bash
# Full pipeline with customer-level aggregation (recommended)
python pipeline_v2_real_data.py

# Standalone version (without aggregation, for comparison)
python credit_score_improved.py data/train.csv
```

---

## Requirements

```
Python >= 3.11
scikit-learn >= 1.4
xgboost >= 2.0
imbalanced-learn >= 0.12
optuna >= 3.5
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
```

Install all dependencies:

```bash
pip install scikit-learn xgboost imbalanced-learn optuna pandas numpy matplotlib
```

---

## Pipeline Description

### Stage 1: Data Loading & Customer Aggregation

The raw dataset (100,000 rows × 28 columns) is cleaned and aggregated:

- **Numeric corruption** — Regex removes embedded underscores/special characters
- **String parsing** — `Credit_History_Age` ("22 Years and 1 Months") → integer months
- **Corruption filtering** — 7,600 invalid `Payment_Behaviour` entries reclassified
- **Customer aggregation** — 8 monthly rows per customer collapsed to 1 using median (numeric) and mode (categorical)
- **Result:** 12,500 genuine customer-level records

### Stage 2: Feature Engineering (52 features)

Six domain-informed financial ratios:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| Debt-to-Income | Outstanding_Debt / Annual_Income | Borrower leverage |
| EMI-to-Income | (Total_EMI × 12) / Annual_Income | Loan commitment burden |
| Delay Rate | Num_Delayed_Payment / Credit_History_Months | Normalized delinquency |
| Investment Ratio | (Amount_invested × 12) / Annual_Income | Financial discipline |
| Utilization × Delay | Credit_Utilization × Delay_from_due | Interaction feature |
| Income per Loan | Annual_Income / Num_of_Loan | Income adequacy |

Plus: within-customer standard deviations (behavioral stability) and maximum delays (worst-case behavior).

### Stage 3: SMOTEENN Resampling

Applied **after** 80/20 stratified split (on training data only):

- **SMOTE** — Generates synthetic minority samples via 5-NN interpolation
- **ENN** — Removes noisy boundary samples misclassified by 3 nearest neighbors
- **Result:** 7,452 balanced training samples

### Stage 4: Model Training & Bayesian Optimization

Three models trained with Optuna TPE hyperparameter search:

| Model | Key Hyperparameters | Optimization |
|-------|-------------------|-------------|
| **XGBoost** | max_depth=9, lr=0.083, n_est=200 | 15 Optuna trials |
| **Random Forest** | max_depth=24, n_est=300, max_feat=log2 | 12 Optuna trials |
| **Stacking Ensemble** | XGB + RF → Logistic Regression meta | 3-fold OOF |

### Stage 5: Evaluation

- 20% stratified holdout (2,500 customers)
- 5-fold stratified cross-validation
- Metrics: Accuracy, Macro-Precision, Macro-Recall, Macro-F1, Weighted-F1, ROC-AUC (OvR)

---

## Results

### Holdout Test (2,500 customers)

```
XGBoost (Optuna)    | Acc=74.24% | Prec=73.41% | Rec=77.02% | F1=73.86% | AUC=87.56%
Random Forest       | Acc=73.84% | Prec=73.17% | Rec=76.78% | F1=73.49% | AUC=86.22%
Stacking (XGB+RF)   | Acc=74.08% | Prec=73.35% | Rec=77.02% | F1=73.75% | AUC=86.45%
```

### Comparison with Original Paper

```
Model                 Acc      F1      AUC      Time
─────────────────────────────────────────────────────
WOA-SVM [original]    80%      80%     N/A      7,422s
GA-SVM  [original]    81%      80%     N/A      5,068s
WOA-LSTM [original]   74%      73%     N/A      39,572s
─────────────────────────────────────────────────────
XGBoost [ours]        74.24%   73.86%  87.56%   386s
Stacking [ours]       74.08%   73.75%  86.45%   386s
─────────────────────────────────────────────────────
The accuracy gap reflects data leakage correction, NOT inferior modeling.
Our pipeline is 103× faster than WOA-LSTM.
```

### Top Features (XGBoost Gain)

```
 1. Credit_Mix                30.41%  ██████████████████████████████
 2. Outstanding_Debt          15.93%  ████████████████
 3. Payment_of_Min_Amount     14.04%  ██████████████
 4. Interest_Rate              6.06%  ██████
 5. Num_of_Delayed_Payment     1.47%  █
 6. Delay_from_due_date        1.43%  █
 7. Num_Credit_Inquiries       1.26%  █
 8. Changed_Credit_Limit       1.21%  █
 9. Debt_to_Income*            1.07%  █
10. Util_x_Delay*              1.06%  █
    * = engineered features
```

---

## Reproducing Results

All experiments use `random_state=42` for full reproducibility.

```bash
# Run the complete pipeline
python pipeline_v2_real_data.py

# Expected output:
# - Console: all metrics, confusion matrices, feature importances
# - File: results.json with all hyperparameters and scores
```

**Hardware used:** 2-core CPU, 4GB RAM (deliberately modest to demonstrate practicality).

---

## Citation

If you use this code or build upon this work, please cite:

```bibtex
@inproceedings{isaku2026credit,
  author    = {Isaku, Ajete},
  title     = {Improved Credit Score Classification Using Ensemble Learning 
               with Bayesian Hyperparameter Optimization},
  booktitle = {Proceedings of the University Conference},
  year      = {2026},
  institution = {University of Pristina}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Dataset by [Rohan Paris](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) on Kaggle
- Original study: "Improving Credit Score Classification Using LSTM and SVM Tuned with WOA" (IEEE, 2024)
- Built with [XGBoost](https://xgboost.readthedocs.io/), [Optuna](https://optuna.org/), [imbalanced-learn](https://imbalanced-learn.org/), [scikit-learn](https://scikit-learn.org/)
