# Opportunity Participation Prediction

A machine-learning pipeline for predicting participation probability from opportunity-related data.

The project uses a **Random Forest classifier** with preprocessing for categorical and numerical features, followed by model evaluation and visual analysis.

## Features

* Automatic identification of the target column
* Date-based train/test splitting when date information is available
* Frequency encoding for categorical features
* Target encoding for high-cardinality categorical features
* KNN-based numerical imputation
* Random Forest classification
* Out-of-sample prediction probabilities
* Confusion matrix generation
* Feature-importance visualization
* Feature-based probability heatmap
* Excel export of predictions

## Data Pipeline

```text
Raw Dataset
    ↓
Target Identification
    ↓
Train/Test Split
    ↓
Feature Encoding
    ↓
Missing-Value Imputation
    ↓
Random Forest
    ↓
Predictions & Evaluation
```

Preprocessing is fitted **only on the training data** to prevent information from the test set leaking into the model.

## Requirements

* Python 3.x
* pandas
* NumPy
* scikit-learn
* Matplotlib
* Seaborn
* openpyxl

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Place the input dataset in the project directory as:

```text
opportunity_data.xlsx
```

Then run:

```bash
python Code_5_refactored_clean.py
```

## Outputs

The pipeline generates:

| File                           | Description                                                               |
| ------------------------------ | ------------------------------------------------------------------------- |
| `opportunity_predictions.xlsx` | Dataset containing predicted participation probabilities                  |
| `feature_importance.png`       | Top Random Forest feature importances                                     |
| `confusion_matrix.png`         | Test-set confusion matrix                                                 |
| `probability_heatmap.png`      | Predicted probability distribution across the two most important features |

## Model

The current implementation uses:

**Random Forest Classifier**

with:

* 100 trees
* fixed random seed (`42`)

Model evaluation is performed on the held-out test set.

## Data Leakage Prevention

The preprocessing pipeline is designed so that statistics and encoders are learned from the **training set only**.

In particular:

* Frequency encodings are calculated from training data.
* Target encoding is fitted using training data.
* Numerical imputation is fitted using training data.
* Test data is only transformed using already-fitted preprocessing steps.

This prevents information from the test set from influencing model training.

## Project Structure

```text
.
├── Code_5_refactored_clean.py
├── opportunity_data.xlsx
├── requirements.txt
├── opportunity_predictions.xlsx
├── feature_importance.png
├── confusion_matrix.png
└── probability_heatmap.png
```

## Notes

The model is intended for experimentation and predictive analysis. Prediction probabilities should not automatically be interpreted as calibrated real-world probabilities without additional validation.

The input dataset is expected to contain a suitable target column and feature columns compatible with the preprocessing pipeline.
