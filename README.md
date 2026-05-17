<h1 align="center">Marketing Customer Segmentation & Churn Prediction</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E.svg"/>
  <img src="https://img.shields.io/badge/XGBoost-classification-EC4E20.svg"/>
  <img src="https://img.shields.io/badge/Streamlit-dashboard-FF4B4B.svg"/>
</p>

<p align="center">
  <b>Customer analytics pipeline: RFM segmentation, K-Means clustering, churn-risk modeling utilities, and an interactive Streamlit dashboard for marketing decision support.</b>
</p>

---

## Business Problem

Marketing teams need to:
- **Identify valuable customers** to focus retention campaigns
- **Predict churn** before customers leave
- **Segment audiences** for personalized marketing strategies

This project provides a **data-driven solution** for public or anonymized e-commerce transaction data.

## What This Project Demonstrates

- Transaction cleaning and customer-level feature engineering
- RFM segmentation and K-Means clustering
- Churn label creation from recency behavior
- Model comparison for Logistic Regression, Random Forest, and optional XGBoost
- Streamlit dashboard for segment exploration and churn-risk review

The dashboard includes generated demo customers so the interface can be shown without exposing private data. Model metrics should be regenerated after connecting a real or public transaction dataset.

## Pipeline

```
1. Data Loading          → UCI Online Retail II dataset (1M+ transactions)
2. Data Cleaning         → Missing values, negative quantities, outliers
3. RFM Feature Engineering → Recency, Frequency, Monetary per customer
4. Customer Segmentation → K-Means clustering (Elbow + Silhouette)
5. Churn Labeling        → Rule-based: no purchase in last 90 days
6. Feature Engineering   → 15+ features: RFM + behavioral + temporal
7. Model Training        → LR, RF, XGBoost with cross-validation
8. Model Evaluation      → ROC-AUC, precision-recall, feature importance
9. Dashboard             → Interactive Streamlit app for exploration
```

## Structure

```
marketing-customer-segmentation/
├── README.md
├── requirements.txt
├── src/
│   ├── data_processing.py             # Cleaning & feature engineering
│   ├── segmentation.py                # RFM + K-Means pipeline
│   └── churn_model.py                 # Training & evaluation
├── app/
│   └── dashboard.py                   # Streamlit dashboard
└── figures/                           # Optional screenshots/plots
```

## Quick Start

```bash
git clone https://github.com/llllenah/marketing-customer-segmentation.git
cd marketing-customer-segmentation
pip install -r requirements.txt

# Launch dashboard
streamlit run app/dashboard.py
```

## Suggested Dataset

For a public-data version, connect the pipeline to [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) or another anonymized transaction dataset.

## Tech Stack

`Python` · `scikit-learn` · `XGBoost` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Plotly` · `Streamlit`

## Author

**Olena Serhiienko** — [@llllenah](https://github.com/llllenah)
