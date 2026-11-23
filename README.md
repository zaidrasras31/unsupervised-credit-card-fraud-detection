# Credit Card Fraud Detection with Anomaly Detection

This Kaggle project compares **Isolation Forest, One-Class SVM, and a PyTorch autoencoder** on a credit card dataset with a raw fraud rate of **0.172%**. The autoencoder achieved the highest test Average Precision (**0.440318**) and F1 score (**0.414141**).

[Open the Kaggle notebook](https://www.kaggle.com/code/zaidrasras/credit-card/notebook) · [Download the dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Dataset and exploratory analysis

The dataset contains 284,807 transactions with 30 input columns (`Time`, `Amount`, and PCA components `V1`–`V28`) plus the binary target `Class` (0 = normal, 1 = fraud). There were no missing values. Removing 1,081 exact duplicate rows left **283,726 transactions**, including **473 frauds** and **283,253 normal transactions**. The cleaned fraud rate was **0.167%**, or about one fraud for every 599 normal transactions.

- Compared class counts on a logarithmic scale and examined transaction amounts with a 99th-percentile cap for visualization and a `log1p` transformation. Fraud transactions had a lower median amount (9.82 vs. 22.00) and a higher mean (123.87 vs. 88.41).
- Compared transaction volume and fraud rates by relative hour over the roughly 48-hour observation window.
- Compared PCA feature distributions by class. The six largest median shifts, normalized by the normal-class interquartile range, were in `V14`, `V17`, `V12`, `V10`, `V16`, and `V3`. All 28 PCA components were retained for modeling.

## Preprocessing and training

The final 31 features were `V1`–`V28`, `Amount_log = log1p(Amount)`, and the sine and cosine of the relative hour. The notebook dropped raw `Time`, raw `Amount`, the helper `Hour` column, and `Class` from the model inputs. It fitted `RobustScaler` on training features and applied the fitted scaler to validation and test features.

| Stratified split (seed 42) | Transactions | Frauds |
| --- | ---: | ---: |
| Train (70%) | 198,608 | 331 |
| Validation (15%) | 42,559 | 71 |
| Test (15%) | 42,559 | 71 |

The notebook uses labels for descriptive analysis, stratification, selecting normal training records for two models, validation threshold selection, and test evaluation. Isolation Forest fits the full training feature matrix. One-Class SVM and the autoencoder use **normal-only novelty detection**, with `Class` used to select their training records. Each model fits transaction features as its inputs.

| Model | Training data | Configuration and anomaly score |
| --- | --- | --- |
| Isolation Forest | All 198,608 training transactions | 300 trees, `max_samples=256`, `contamination=0.0017`, seed 42. Score is the negated decision function. |
| One-Class SVM | 30,000 sampled normal transactions, seed 42 | RBF kernel, `gamma="scale"`, `nu=0.005`. Score is the negated decision function. |
| PyTorch autoencoder | All 198,277 normal training transactions | Architecture `31 → 20 → 10 → 6 → 10 → 20 → 31` with LeakyReLU and batch normalization. Trained for 20 epochs with MSE loss, AdamW (learning rate 0.001, weight decay 0.00001), batch size 512, and seed 42. Score is per-transaction reconstruction MSE. |

## Test results

Higher scores indicate more anomalous transactions. For each model, the notebook selected the score threshold that maximized validation F1 and applied that threshold to the held-out test set. Average Precision (labeled PR-AUC in the notebook) and ROC-AUC use continuous scores. Precision, recall, and F1 use the selected threshold.

| Model | Average Precision | ROC-AUC | Validation threshold | Test precision | Test recall | Test F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch autoencoder | **0.440318** | 0.957829 | 5.717340 | **0.322835** | 0.577465 | **0.414141** |
| Isolation Forest | 0.196205 | **0.960068** | -0.022894 | 0.168750 | 0.380282 | 0.233766 |
| One-Class SVM | 0.145414 | 0.949918 | 0.311518 | 0.164336 | **0.661972** | 0.263305 |

The test fraud rate gives a random-ranking Average Precision baseline of about **0.001668**. The autoencoder flagged **41 of 71 frauds**, missed **30**, and produced **86 false alerts**. One-Class SVM had the highest recall, while Isolation Forest had the highest ROC-AUC. The notebook includes precision–recall curves, ROC curves, confusion matrices, and an analysis of the frauds the autoencoder caught and missed.

## Tech stack

Python 3.13, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, and PyTorch. The saved autoencoder run used CPU. The notebook can use CUDA when available.

## Project files

| File | Contents |
| --- | --- |
| `credit-card.ipynb` | Data analysis, preprocessing, training, threshold selection, and evaluation |
| `README.md` | Project documentation and run instructions |

## How to run

**On Kaggle:** Open the [notebook](https://www.kaggle.com/code/zaidrasras/credit-card/notebook), ensure the Credit Card Fraud Detection dataset is attached, and run all cells from top to bottom.

**Locally:**

1. Clone the repository using the URL from its GitHub **Code** menu, or download the notebook from Kaggle.
2. Install the dependencies.

   ```bash
   python -m pip install numpy pandas matplotlib seaborn scikit-learn torch notebook
   ```

3. Download `creditcard.csv` from the [dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it next to the notebook. Change the Kaggle input path in the data-loading cell to `pd.read_csv("creditcard.csv")`.
4. Open the notebook in Jupyter Notebook or VS Code and run all cells from top to bottom.
