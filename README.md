# Unsupervised Credit Card Fraud Detection (Anomaly Detection)

An end-to-end machine learning and deep learning pipeline for detecting anomalous credit card transactions under extreme class imbalance. The raw fraud rate is **0.172%**.

**Kaggle Notebook** · [Unsupervised Credit Card Fraud Detection](https://www.kaggle.com/code/zaidrasras/credit-card/notebook)

## 🎯 Project Objective & Unsupervised Constraint

The models learn from transaction features with `Class` excluded from preprocessing, scaling, feature selection, and training. Ground-truth labels (`0` for normal and `1` for fraud) are used for descriptive class comparisons and post-training evaluation. Use validation labels for any label-based threshold selection and reserve test labels for final evaluation.

## 📊 Dataset & Cleaning Pipeline

- **Source** · [Credit Card Fraud Detection Dataset (MLG–ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Raw data** · 284,807 transactions and 31 columns, including `Time`, PCA-transformed `V1`–`V28`, `Amount`, and the target `Class`.
- **Missing values** · 0.
- **Deduplication** · Removed 1,081 exact duplicate rows, leaving 283,726 transactions (283,253 normal and 473 fraudulent, a **0.167%** fraud rate).
- **Evaluation strategy** · Fit preprocessing and models on training features, use validation data for tuning, and report final metrics on held-out test data.

## 🔍 Exploratory Data Analysis & Feature Engineering

1. **Class imbalance** · Visualized class counts on a logarithmic scale. After deduplication, there was approximately one fraudulent transaction for every 599 normal transactions.
2. **Transaction amount** · Compared raw amounts, distributions capped at the 99th percentile, and `log1p(Amount)`. The reported fraud median was 9.82 versus 22.00 for normal transactions. The reported fraud mean was 123.87 versus 88.41 for normal transactions.
3. **Time patterns** · Derived an hour index with `(Time // 3600) % 24` and compared transaction volume with fraud counts across the 48-hour observation window. The index is relative to the first recorded transaction.
4. **Feature scaling** · Used `RobustScaler` or `StandardScaler` for amount and time-derived features, depending on the model pipeline.

## 🧠 Unsupervised Models & Hyperparameter Tuning

| Model | How it scores anomalies | Parameters explored |
| --- | --- | --- |
| Isolation Forest | Transactions isolated in fewer tree splits receive higher anomaly scores. | `n_estimators`, `max_samples`, `contamination` |
| One-Class SVM | An RBF kernel estimates a boundary around the training distribution. Transactions outside the boundary are flagged. | `nu`, `gamma` |
| Deep Autoencoder | A bottleneck network learns to reconstruct its inputs using mean squared error. Large reconstruction errors flag unusual transactions. | Reconstruction-error percentile |

## 📈 Results & Model Comparison

An all-normal prediction would achieve about **99.83% accuracy** on the cleaned dataset. The comparison emphasizes **Average Precision (AP)**, ROC-AUC, precision, recall, and F1. AP summarizes precision and recall across score thresholds. Precision, recall, and F1 in the table refer to the chosen decision threshold.

| Model | ROC-AUC | Average Precision | Precision | Recall | F1 | Threshold or settings from draft |
| --- | --- | --- | --- | --- | --- | --- |
| Isolation Forest | — | — | — | — | — | `contamination=0.0017` |
| One-Class SVM | — | — | — | — | — | `kernel='rbf'` |
| Deep Autoencoder | — | — | — | — | — | 99.8th percentile of reconstruction MSE |

The draft did not include final metric values. See the [Kaggle notebook](https://www.kaggle.com/code/zaidrasras/credit-card/notebook) for the evaluation cells. The decision threshold controls the trade-off between finding more fraud and sending more transactions for review.

## 🛠️ Tech Stack

- **Language** · Python
- **Data analysis** · NumPy and pandas
- **Visualization** · Matplotlib and Seaborn
- **Modeling** · scikit-learn and PyTorch

## 📁 Project Structure

| File | Purpose |
| --- | --- |
| `credit-card.ipynb` | EDA, preprocessing, model training, tuning, and evaluation |
| `README.md` | Project overview and setup instructions |
| `requirements.txt` | Dependency list, if included in the repository |

## 🚀 How to Run

1. **Clone the repository.** Replace the placeholder URL with your actual GitHub repository URL.

   ```bash
   git clone https://github.com/your-username/unsupervised-credit-card-fraud-detection.git
   cd unsupervised-credit-card-fraud-detection
   ```

2. **Install dependencies.**

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn torch
   ```

3. **Download the dataset.** Get `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root. If the notebook uses a Kaggle-specific input path, change it to `creditcard.csv` for a local run.

4. **Run the notebook.** Open `credit-card.ipynb` in Jupyter Notebook or VS Code and run all cells from top to bottom. On Kaggle, open the linked notebook with the dataset attached and run all cells there.

### Quick terminal command to create `README.md`

From the repository root, run the following line in Bash. It will overwrite an existing `README.md`.

```bash
cat << 'EOF' > README.md
```

Paste the Markdown content, press Enter, type `EOF` on its own line, and press Enter again.
