"""
Interactive Streamlit dashboard for marketing customer segmentation analysis.

Run: streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1F4E79, #2ECC71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .segment-champion { border-left: 5px solid #FFD700; padding-left: 1rem; }
    .segment-loyal { border-left: 5px solid #2ECC71; padding-left: 1rem; }
    .segment-risk { border-left: 5px solid #E67E22; padding-left: 1rem; }
    .segment-dormant { border-left: 5px solid #95A5A6; padding-left: 1rem; }
</style>
""", unsafe_allow_html=True)


def generate_demo_data(n_customers: int = 2000) -> pd.DataFrame:
    """Generate realistic demo customer data for dashboard demonstration."""
    np.random.seed(42)

    segments = np.random.choice(
        ["🏆 Champions", "💚 Loyal", "🔄 At Risk", "😴 Dormant"],
        size=n_customers,
        p=[0.18, 0.32, 0.28, 0.22],
    )

    data = []
    for i, seg in enumerate(segments):
        if seg == "🏆 Champions":
            rec, freq, mon = np.random.exponential(15), np.random.poisson(12) + 1, np.random.exponential(4200)
            churn_prob = 0.05
        elif seg == "💚 Loyal":
            rec, freq, mon = np.random.exponential(40), np.random.poisson(6) + 1, np.random.exponential(1800)
            churn_prob = 0.15
        elif seg == "🔄 At Risk":
            rec, freq, mon = np.random.exponential(120), np.random.poisson(2) + 1, np.random.exponential(950)
            churn_prob = 0.55
        else:
            rec, freq, mon = np.random.exponential(200), max(1, np.random.poisson(1)), np.random.exponential(320)
            churn_prob = 0.85

        data.append({
            "CustomerID": 10000 + i,
            "Recency": max(1, int(rec)),
            "Frequency": max(1, int(freq)),
            "Monetary": round(max(10, mon), 2),
            "Segment_Label": seg,
            "Churn_Probability": min(0.99, max(0.01, churn_prob + np.random.normal(0, 0.1))),
            "Country": np.random.choice(["UK", "Germany", "France", "Spain", "Netherlands"], p=[0.5, 0.15, 0.15, 0.1, 0.1]),
            "Avg_Order_Value": round(max(5, mon / max(1, freq) + np.random.normal(0, 20)), 2),
        })

    return pd.DataFrame(data)


def main():
    st.markdown('<h1 class="main-header">📊 Customer Segmentation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Load demo data
    df = generate_demo_data()

    # ─── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🔧 Filters")

        selected_segments = st.multiselect(
            "Customer Segments",
            options=df["Segment_Label"].unique(),
            default=df["Segment_Label"].unique(),
        )

        selected_countries = st.multiselect(
            "Countries",
            options=sorted(df["Country"].unique()),
            default=sorted(df["Country"].unique()),
        )

        monetary_range = st.slider(
            "Revenue Range ($)",
            min_value=0,
            max_value=int(df["Monetary"].max()),
            value=(0, int(df["Monetary"].max())),
        )

        churn_threshold = st.slider(
            "Churn Risk Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )

    # Apply filters
    filtered = df[
        (df["Segment_Label"].isin(selected_segments))
        & (df["Country"].isin(selected_countries))
        & (df["Monetary"].between(*monetary_range))
    ]

    # ─── KPI Metrics ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Total Customers", f"{len(filtered):,}")
    with col2:
        st.metric("💰 Total Revenue", f"${filtered['Monetary'].sum():,.0f}")
    with col3:
        st.metric("📦 Avg Order Value", f"${filtered['Avg_Order_Value'].mean():,.0f}")
    with col4:
        churn_rate = (filtered["Churn_Probability"] > churn_threshold).mean() * 100
        st.metric("⚠️ Churn Risk Rate", f"{churn_rate:.1f}%")

    st.markdown("---")

    # ─── Segmentation Overview ────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🎯 Segment Distribution")
        seg_counts = filtered["Segment_Label"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        color_map = {
            "🏆 Champions": "#FFD700",
            "💚 Loyal": "#2ECC71",
            "🔄 At Risk": "#E67E22",
            "😴 Dormant": "#95A5A6",
        }
        fig_pie = px.pie(
            seg_counts, values="Count", names="Segment",
            color="Segment", color_discrete_map=color_map,
            hole=0.4,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("💰 Revenue by Segment")
        seg_revenue = (
            filtered.groupby("Segment_Label")["Monetary"]
            .agg(["sum", "mean", "count"])
            .reset_index()
        )
        seg_revenue.columns = ["Segment", "Total Revenue", "Avg Revenue", "Customers"]
        fig_bar = px.bar(
            seg_revenue, x="Segment", y="Total Revenue",
            color="Segment", color_discrete_map=color_map,
            text="Customers",
        )
        fig_bar.update_traces(texttemplate="%{text} customers", textposition="outside")
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ─── Scatter Plot ─────────────────────────────────────────────────────
    st.subheader("🔍 Customer Landscape")
    fig_scatter = px.scatter(
        filtered, x="Recency", y="Monetary",
        color="Segment_Label", size="Frequency",
        color_discrete_map=color_map,
        hover_data=["CustomerID", "Frequency", "Avg_Order_Value"],
        opacity=0.6,
        labels={"Recency": "Recency (days)", "Monetary": "Total Revenue ($)"},
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ─── Churn Risk Analysis ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚠️ Churn Risk Analysis")

    col_churn1, col_churn2 = st.columns(2)

    with col_churn1:
        fig_hist = px.histogram(
            filtered, x="Churn_Probability", color="Segment_Label",
            color_discrete_map=color_map, nbins=30,
            barmode="overlay", opacity=0.7,
            labels={"Churn_Probability": "Churn Probability"},
        )
        fig_hist.add_vline(x=churn_threshold, line_dash="dash", line_color="red",
                          annotation_text=f"Threshold: {churn_threshold:.0%}")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_churn2:
        at_risk = filtered[filtered["Churn_Probability"] > churn_threshold]
        st.markdown(f"### 🚨 {len(at_risk)} customers at risk")
        st.markdown(f"**Revenue at risk:** ${at_risk['Monetary'].sum():,.0f}")

        if len(at_risk) > 0:
            st.dataframe(
                at_risk.nlargest(10, "Monetary")[
                    ["CustomerID", "Segment_Label", "Monetary", "Recency", "Churn_Probability"]
                ].style.format({
                    "Monetary": "${:,.0f}",
                    "Churn_Probability": "{:.0%}",
                }),
                use_container_width=True,
            )

    # ─── Marketing Recommendations ───────────────────────────────────────
    st.markdown("---")
    st.subheader("💡 Marketing Recommendations")

    rec_col1, rec_col2, rec_col3, rec_col4 = st.columns(4)

    with rec_col1:
        st.markdown("#### 🏆 Champions")
        n_champ = len(filtered[filtered["Segment_Label"] == "🏆 Champions"])
        st.markdown(f"**{n_champ}** customers")
        st.markdown("- VIP loyalty programs\n- Early access to products\n- Referral incentives")

    with rec_col2:
        st.markdown("#### 💚 Loyal")
        n_loyal = len(filtered[filtered["Segment_Label"] == "💚 Loyal"])
        st.markdown(f"**{n_loyal}** customers")
        st.markdown("- Cross-sell campaigns\n- Upsell higher tiers\n- Membership offers")

    with rec_col3:
        st.markdown("#### 🔄 At Risk")
        n_risk = len(filtered[filtered["Segment_Label"] == "🔄 At Risk"])
        st.markdown(f"**{n_risk}** customers")
        st.markdown("- Re-engagement emails\n- Personalized discounts\n- Win-back campaigns")

    with rec_col4:
        st.markdown("#### 😴 Dormant")
        n_dorm = len(filtered[filtered["Segment_Label"] == "😴 Dormant"])
        st.markdown(f"**{n_dorm}** customers")
        st.markdown("- Final win-back offer\n- Survey: why left?\n- Consider de-prioritizing")


if __name__ == "__main__":
    main()
