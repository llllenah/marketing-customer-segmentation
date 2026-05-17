"""
Customer segmentation using RFM analysis and K-Means clustering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def find_optimal_k(
    rfm_features: pd.DataFrame,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> tuple[list, list]:
    """
    Determine optimal number of clusters using Elbow and Silhouette methods.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_features)

    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        print(f"  k={k:2d}  inertia={km.inertia_:,.0f}  silhouette={silhouettes[-1]:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(list(k_range), inertias, "o-", color="#1F4E79", linewidth=2)
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")

    axes[1].plot(list(k_range), silhouettes, "s-", color="#2CA02C", linewidth=2)
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Analysis")

    plt.suptitle("Optimal Cluster Selection", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    return inertias, silhouettes


def segment_customers(
    rfm: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Perform K-Means segmentation on RFM features.
    
    Returns rfm DataFrame with added 'Segment' and 'Segment_Label' columns.
    """
    feature_cols = ["Recency", "Frequency", "Monetary"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[feature_cols])

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm["Segment"] = km.fit_predict(X_scaled)

    # Label segments by average monetary value
    segment_order = (
        rfm.groupby("Segment")["Monetary"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    label_map = {}
    labels = ["🏆 Champions", "💚 Loyal", "🔄 At Risk", "😴 Dormant"]
    for i, seg in enumerate(segment_order):
        label_map[seg] = labels[min(i, len(labels) - 1)]

    rfm["Segment_Label"] = rfm["Segment"].map(label_map)

    # Summary
    summary = (
        rfm.groupby("Segment_Label")
        .agg(
            Count=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
        )
        .round(1)
    )
    summary["Pct"] = (summary["Count"] / summary["Count"].sum() * 100).round(1)

    print("\n📊 Customer Segments:")
    print(summary.to_string())

    return rfm


def plot_segments(rfm: pd.DataFrame) -> None:
    """Visualize customer segments with multiple plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    palette = {"🏆 Champions": "#FFD700", "💚 Loyal": "#2ECC71", "🔄 At Risk": "#E67E22", "😴 Dormant": "#95A5A6"}

    # Scatter: Frequency vs Monetary
    for label, group in rfm.groupby("Segment_Label"):
        axes[0, 0].scatter(
            group["Frequency"], group["Monetary"],
            label=label, alpha=0.5, s=20,
            color=palette.get(label, "#999"),
        )
    axes[0, 0].set_xlabel("Frequency (orders)")
    axes[0, 0].set_ylabel("Monetary ($)")
    axes[0, 0].set_title("Frequency vs Monetary by Segment")
    axes[0, 0].legend()

    # Scatter: Recency vs Monetary
    for label, group in rfm.groupby("Segment_Label"):
        axes[0, 1].scatter(
            group["Recency"], group["Monetary"],
            label=label, alpha=0.5, s=20,
            color=palette.get(label, "#999"),
        )
    axes[0, 1].set_xlabel("Recency (days)")
    axes[0, 1].set_ylabel("Monetary ($)")
    axes[0, 1].set_title("Recency vs Monetary by Segment")
    axes[0, 1].legend()

    # Bar: Segment sizes
    segment_counts = rfm["Segment_Label"].value_counts()
    colors = [palette.get(s, "#999") for s in segment_counts.index]
    axes[1, 0].bar(segment_counts.index, segment_counts.values, color=colors)
    axes[1, 0].set_title("Segment Distribution")
    axes[1, 0].set_ylabel("Number of Customers")
    plt.setp(axes[1, 0].get_xticklabels(), rotation=15)

    # Box: Revenue by segment
    order = ["🏆 Champions", "💚 Loyal", "🔄 At Risk", "😴 Dormant"]
    existing_order = [o for o in order if o in rfm["Segment_Label"].values]
    sns.boxplot(
        data=rfm, x="Segment_Label", y="Monetary",
        order=existing_order, ax=axes[1, 1],
        palette=palette, showfliers=False,
    )
    axes[1, 1].set_title("Revenue Distribution by Segment")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Total Revenue ($)")
    plt.setp(axes[1, 1].get_xticklabels(), rotation=15)

    plt.suptitle("Customer Segmentation Analysis", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.show()
