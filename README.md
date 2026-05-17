# Marketing Customer Segmentation and Churn Prediction

Customer analytics pipeline for RFM segmentation, K-Means clustering, churn-risk modeling, and an interactive Streamlit dashboard.

## Business Problem

Marketing and CRM teams often need to identify high-value customers, estimate churn risk, and translate customer behavior into targeted retention or reactivation campaigns. This project demonstrates a reusable analytics workflow for public or anonymized e-commerce transaction data.

## What This Project Demonstrates

- Transaction cleaning and customer-level feature engineering.
- RFM segmentation and K-Means clustering.
- Rule-based churn labeling from recency behavior.
- Model comparison for Logistic Regression, Random Forest, and optional XGBoost.
- Streamlit dashboard for customer filtering, segment exploration, and churn-risk review.

The dashboard includes generated demo customers so the interface can be shown without exposing private data. Model metrics should be regenerated after connecting a real or public transaction dataset.

## Pipeline

```text
1. Data loading              Public or anonymized transaction data
2. Data cleaning             Missing values, negative quantities, outliers
3. RFM features              Recency, frequency, monetary value
4. Segmentation              K-Means clustering with elbow/silhouette checks
5. Churn labeling            Rule-based inactivity threshold
6. Behavioral features       Tenure, AOV, product count, revenue trend
7. Model training            Logistic Regression, Random Forest, XGBoost
8. Evaluation                ROC-AUC, precision/recall, feature importance
9. Dashboard                 Interactive Streamlit app
```

## Repository Structure

```text
marketing-customer-segmentation/
├── README.md
├── requirements.txt
├── src/
│   ├── data_processing.py
│   ├── segmentation.py
│   └── churn_model.py
└── app/
    └── dashboard.py
```

## Quick Start

```bash
git clone https://github.com/llllenah/marketing-customer-segmentation.git
cd marketing-customer-segmentation
pip install -r requirements.txt
streamlit run app/dashboard.py
```

## Suggested Dataset

For a public-data version, connect the pipeline to [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) or another anonymized transaction dataset.

## Tech Stack

Python, scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Streamlit.
