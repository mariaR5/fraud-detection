# Fraud Detection using Simulated Transactions

## Dataset

This project uses a **simulated dataset** of financial transactions, containing both original and fraudulent samples. Fraud is generated based on **three scenarios**:

1. **Large Amount Fraud**

   * Any transaction with **amount > 220** is labeled as fraud.
   * Feature engineered:

     * `IS_LARGE_AMT` → Binary flag for large transactions.

2. **Terminal-Based Fraud**

   * Each day, two terminals are randomly chosen; **all transactions in the next 28 days** on these terminals are fraudulent.
   * Features engineered:

     * `TERMINAL_FRAUD_COUNT_7D`, `TERMINAL_FRAUD_COUNT_28D` → Count of past frauds at a terminal.
     * `TERMINAL_FRAUD_RATE_7D`, `TERMINAL_FRAUD_RATE_28D` → Fraud rate at a terminal.

3. **Customer Behaviour Fraud**

   * Randomly chosen customers have **1/3 of their transactions multiplied by 5** over 14 days and marked fraudulent.
   * Features engineered:

     * `CUSTOMER_FRAUD_COUNT_7D`, `CUSTOMER_FRAUD_COUNT_14D` → Fraud counts per customer.
     * `CUSTOMER_FRAUD_RATE_7D`, `CUSTOMER_FRAUD_RATE_14D` → Fraud rate per customer.
     * `CUSTOMER_AVG_AMOUNT_7D`, `CUSTOMER_AVG_AMOUNT_RATIO` → Rolling average spend and deviation ratio.
     * `IS_SPIKE_SPENDING` → Flag for spending spike (transaction > 4× rolling average).

## Additional Features

* **Time-based features**: Hour of transaction, day of week, morning/afternoon/evening/night indicators, peak-hour flag.

---

## Model Training

* Algorithm: **XGBoost Classifier**
* Train-test split: **Time-based (70%-30%)**
* Class imbalance handled with: `scale_pos_weight ≈ 120`
* Feature selection: Less important features removed for optimized training

---

## Results

### Base Model (all features)

* **Recall (Fraud)**: 0.94
* **Precision (Fraud)**: 0.33
* **F1 (Fraud)**: 0.49

### Selected Features Model

* **Recall (Fraud)**: 0.93
* **Precision (Fraud)**: 0.41
* **F1 (Fraud)**: 0.57

### After Hyperparameter Tuning

* **Recall (Fraud)**: 0.95
* **Precision (Fraud)**: 0.26
* **F1 (Fraud)**: 0.40

### After Threshold Tuning (Threshold = 0.85)

* **Recall (Fraud)**: 0.89
* **Precision (Fraud)**: 0.78
* **F1 (Fraud)**: 0.83
* **Overall Accuracy**: 1.00

**Final Chosen Model**: XGBoost with selected features, threshold = 0.85

---

## Usage

The final model is saved as:

```bash
xgboost_fraud_detection_model.pkl
```

It can be loaded with:

```python
import joblib
model = joblib.load("xgboost_fraud_detection_model.pkl")
```


