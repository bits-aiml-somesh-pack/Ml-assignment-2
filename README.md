# Bank Marketing — Term Deposit Subscription Classification

Machine Learning Assignment 2 — Multi-model classification with a Streamlit deployment.

## a. Problem Statement

Banks run outbound telemarketing campaigns to persuade clients to subscribe to a **term
deposit**. Calling every client in a large customer base is expensive, so the bank wants
to predict, ahead of time, which clients are most likely to subscribe if contacted. This
is framed as a **binary classification problem**: given a client's demographic, financial,
and campaign-related attributes, predict whether they will subscribe to a term deposit
(`yes`) or not (`no`).

Five supervised classification models are trained and compared on this task, and the
best-performing model is exposed through an interactive Streamlit web application.

## b. Dataset Description

- **Source:** [UCI Machine Learning Repository — Bank Marketing Data Set](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **File used:** `bank-additional-full.csv`
- **Instances:** 41,188 (well above the required minimum of 500)
- **Raw features:** 20 input columns + 1 target (`y`)
- **Target variable:** `y` — has the client subscribed to a term deposit? (`yes` / `no`)
- **Class balance:** Imbalanced — about 11% `yes`, 89% `no`

**Feature groups used for modeling (19 features, above the required minimum of 12):**

| Type | Features |
|---|---|
| Numeric (9) | `age`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` |
| Categorical (10, one-hot encoded) | `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome` |

**Note on `duration`:** The original dataset includes a `duration` column (last call
duration in seconds). This value is only known *after* a call has ended, so it leaks the
outcome and cannot be used for a realistic, pre-call prediction model. It was **dropped**
before training, consistent with the guidance published with this dataset.

**Preprocessing:**
- Numeric features scaled with `StandardScaler`
- Categorical features encoded with `OneHotEncoder` (unknown categories ignored at inference)
- Target label-encoded (`no` → 0, `yes` → 1)
- 80/20 stratified train/test split (`random_state=42`)

## c. GitHub Repository Link

https://github.com/bits-aiml-somesh-pack/Ml-assignment-2

Repository contains:
```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- test_data.csv
│-- data/
│   └-- bank-additional-full.csv
│-- model/
│   │-- train_models.py
│   │-- preprocessor.pkl
│   │-- label_encoder.pkl
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest.pkl
│   └-- metrics.json
```

## d. Models Used

Five classification models were implemented on the same dataset and split:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier (k=15)
4. Naive Bayes Classifier (Gaussian)
5. Random Forest Classifier (Ensemble, 200 trees)

> Note: The assignment brief lists 5 named models but also refers to "6 ML models" while
> the comparison table template itself only has 5 rows. The 5 explicitly named models
> above were implemented; no 6th model was specified anywhere in the brief.

### Comparison Table

Metrics computed on the held-out test set (8,238 rows, 20% of the data):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9009 | 0.8008 | 0.6905 | 0.2188 | 0.3322 | 0.3516 |
| Decision Tree | 0.8418 | 0.6258 | 0.3131 | 0.3384 | 0.3252 | 0.2360 |
| kNN | 0.9003 | 0.7769 | 0.6458 | 0.2554 | 0.3660 | 0.3641 |
| Naive Bayes | 0.8049 | 0.7755 | 0.3172 | 0.6347 | 0.4230 | 0.3490 |
| Random Forest (Ensemble) | 0.9018 | 0.8129 | 0.6809 | 0.2414 | 0.3564 | 0.3665 |

*(Full precision values are stored in `model/metrics.json`.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest precision (0.69) among all models and strong AUC (0.80), but very low recall (0.22) — it is conservative and misses most actual subscribers. Good baseline, biased toward the majority class due to class imbalance. |
| Decision Tree | Weakest model overall — lowest AUC (0.63) and MCC (0.24). A single unpruned tree overfits the training data and generalizes poorly on this imbalanced dataset. |
| kNN | Similar accuracy to Logistic Regression with slightly better recall and the second-highest MCC. Performance is sensitive to feature scaling (handled here via `StandardScaler`) and the choice of k. |
| Naive Bayes | Lowest accuracy (0.80) but by far the highest recall (0.63) — it catches most true subscribers at the cost of many false positives, since the Gaussian independence assumption doesn't hold well for the correlated numeric features. Useful if the business goal is "don't miss a likely subscriber." |
| Random Forest (Ensemble) | Best overall balance — highest MCC (0.368) and AUC (0.79) among all 5 models, with much better generalization than the single Decision Tree due to ensembling/bagging. |
| **Overall Winner** | **Random Forest (Ensemble)** — it achieves the best MCC and a strong AUC, which are the most reliable metrics on this imbalanced dataset (accuracy alone is misleading since ~89% of clients are `no`). |

## Live Streamlit App Link

`<PASTE YOUR DEPLOYED STREAMLIT APP LINK HERE>`

## How to Run Locally

```bash
pip install -r requirements.txt

# Train all models (writes model/*.pkl, model/metrics.json, test_data.csv)
python model/train_models.py

# Launch the Streamlit app
streamlit run app.py
```

Then, in the app sidebar:
1. Upload `test_data.csv` (or any CSV with the same raw feature columns + a `y` column)
2. Select a model from the dropdown
3. View accuracy/AUC/precision/recall/F1/MCC, the confusion matrix, and the classification report
