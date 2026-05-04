import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from plotly.subplots import make_subplots

# =Config
st.set_page_config(
    page_title="BoE MPC Sentiment Tracker",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    scores = pd.read_csv("data/processed/sentiment_scores_full.csv")
    scores["date"] = pd.to_datetime(scores["date"])
    scores = scores.sort_values("date").reset_index(drop=True)

    # Demeaned sentiment
    scores["finbert_net"] = scores["positive"] - scores["negative"]
    rolling_mean = scores["finbert_net"].rolling(window=12, min_periods=6).mean()
    scores["finbert_demeaned"] = scores["finbert_net"] - rolling_mean

    # Calculating polarity
    scores["finbert_polarity_raw"] = (
            (scores["positive"] - scores["negative"]) / (scores["positive"] + scores["negative"])
    )
    # Gate against weakly opinionated meetings
    opinion_mass = scores["positive"] + scores["negative"]
    scores.loc[opinion_mass < 0.15, "finbert_polarity_raw"] = pd.NA

    scores["opinion_mass"] = opinion_mass

    # Bank rate parsing
    bank_rate = pd.read_csv("data/raw/bank_rate.csv")
    bank_rate.columns = ["date", "bank_rate"]
    # Try 4-digit year
    parsed = pd.to_datetime(bank_rate["date"], format="%d %b %Y", errors="coerce")
    if parsed.isna().any():
        # If 4-digit year fails try 2-digit
        parsed = pd.to_datetime(bank_rate["date"], format="%d %b %y", errors="coerce")
    assert parsed.notna().all(), "Some bank rate dates failed to parse"
    bank_rate["date"] = parsed
    bank_rate["bank_rate"] = pd.to_numeric(bank_rate["bank_rate"], errors="coerce")
    bank_rate = bank_rate.dropna().sort_values("date").reset_index(drop=True)

    return scores, bank_rate

scores_df, bank_rate = load_data()

# Header
st.title("Bank of England MPC Sentiment Tracker")
st.markdown(
    "Tracks sentiment in MPC minutes using **FinBERT**, "
    "a BERT model specially trained on financial text. Sentiment is demeaned against "
    "a 12-meeting rolling window so deviations reflect shifts in tone."
)

st.divider()

# Sidebar controls
st.sidebar.header("Controls")

min_date = scores_df["date"].min().date()
max_date = scores_df["date"].max().date()

# Date range filter slider
date_range = st.sidebar.slider(
    "Date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY"
)

# Smoothing window slider
smoothing = st.sidebar.slider(
    "Smoothing window (meetings)",
    min_value=1,
    max_value=6,
    value=3,
    help="Rolling average applied to sentiment score"
)

# Change sentiment metric radio buttons
metric = st.sidebar.radio(
    "Sentiment metric",
    options=["Net Sentiment", "Polarity", "Components"],
    index=0,
    help=(
        "Net Sentiment (pos − neg): positive minus negative share, demeaned against a 12-meeting baseline. "
        "Polarity ((pos − neg) / (pos + neg)): raw directional tilt normalised by opinion mass. "
        "Components: positive and negative shares plotted separately."
    )
)

show_raw = st.sidebar.checkbox("Show unsmoothed sentiment", value=True)

# Opinion mass
show_uncertainty = st.sidebar.checkbox(
    "Overlay opinion mass",
    value=False,
    help=" Sums positive and negative sentiments. Shows how opinionated each meeting was, regardless of direction. Useful for spotting contested meetings."
)

# Filtering for each metric
scores_full = scores_df.copy()

if metric.startswith("Net"):
    scores_full["primary"] = scores_full["finbert_demeaned"]
    primary_label = "Net sentiment (demeaned)"
    y_axis_label = "Sentiment Deviation from 12-meeting Baseline"
elif metric.startswith("Polarity"):
    scores_full["primary"] = scores_full["finbert_polarity_raw"]
    primary_label = "Polarity (raw)"
    y_axis_label = "Polarity (raw)"
else:  # Components
    primary_label = None
    y_axis_label = "Share of sentences"

if metric != "Components":
    scores_full["primary_smooth"] = (
        scores_full["primary"].rolling(window=smoothing, center=True).mean()
    )

mask = (scores_full["date"].dt.date >= date_range[0]) & (scores_full["date"].dt.date <= date_range[1])
filtered = scores_full[mask].copy()

rate_mask = (bank_rate["date"].dt.date >= date_range[0]) & (bank_rate["date"].dt.date <= date_range[1])
filtered_rate = bank_rate[rate_mask]

# Plotting
fig = make_subplots(specs=[[{"secondary_y": True}]])

if metric == "Components":
    fig.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["positive"],
        mode="lines", name="Positive share",
        line=dict(color="seagreen", width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>Positive: %{y:.3f}<extra></extra>"
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["negative"],
        mode="lines", name="Negative share",
        line=dict(color="indianred", width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>Negative: %{y:.3f}<extra></extra>"
    ), secondary_y=False)
else:
    if show_raw:
        fig.add_trace(go.Scatter(
            x=filtered["date"], y=filtered["primary"],
            mode="lines", name=f"Unsmoothed {primary_label.lower()}",
            line=dict(color="steelblue", width=0.8), opacity=0.3,
            hovertemplate="<b>%{x|%b %Y}</b><br>Unsmoothed: %{y:.3f}<extra></extra>"
        ), secondary_y=False)

    smoothed_label = (
        primary_label if smoothing == 1
        else f"{primary_label} ({smoothing}-meeting avg)"
    )
    fig.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["primary_smooth"],
        mode="lines", name=smoothed_label,
        line=dict(color="steelblue", width=2.5),
        hovertemplate="<b>%{x|%b %Y}</b><br>Sentiment: %{y:.3f}<extra></extra>"
    ), secondary_y=False)

if show_uncertainty:
    fig.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["opinion_mass"],
        mode="lines", name="Opinion mass (pos + neg)",
        line=dict(color="orange", width=1.2, dash="dot"),
        opacity=0.6,
        hovertemplate="<b>%{x|%b %Y}</b><br>Opinion mass: %{y:.3f}<extra></extra>"
    ), secondary_y=False)

fig.add_trace(go.Scatter(
    x=filtered_rate["date"], y=filtered_rate["bank_rate"],
    mode="lines", name="Bank Rate (%)",
    line=dict(color="white", width=2, dash="dash"),
    hovertemplate="<b>%{x|%b %Y}</b><br>Bank Rate: %{y:.2f}%<extra></extra>"
), secondary_y=True)

fig.add_hline(y=0, line_dash="dot", line_color="grey", line_width=0.8)

# Axis ranges from filtered data
if metric == "Components":
    sent_series = pd.concat([filtered["positive"], filtered["negative"]]).dropna()
else:
    sent_series = pd.concat([filtered["primary"], filtered["primary_smooth"]]).dropna()

if show_uncertainty:
    sent_series = pd.concat([sent_series, filtered["opinion_mass"].dropna()])

if len(sent_series) > 0:
    sent_min, sent_max = sent_series.min(), sent_series.max()
    sent_pad = max((sent_max - sent_min) * 0.1, 0.05)
    sent_range = [sent_min - sent_pad, sent_max + sent_pad]
else:
    sent_range = [-0.8, 0.4]

if len(filtered_rate) > 0:
    rate_max = filtered_rate["bank_rate"].max()
    rate_range = [0, max(rate_max * 1.15, 1.0)]
else:
    rate_range = [0, 7]

fig.update_yaxes(title_text=y_axis_label, range=sent_range, secondary_y=False)
fig.update_yaxes(title_text="Bank Rate (%)", range=rate_range, showgrid=False, secondary_y=True)

fig.update_layout(
    title="MPC Minutes Sentiment vs Bank Rate",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
    height=550,
    template="plotly_dark",
    xaxis=dict(domain=[0, 0.95])
)

st.plotly_chart(fig, use_container_width=True)

# Data table
with st.expander("View underlying data"):
    display_cols = ["date", "positive", "negative", "neutral",
                    "finbert_net", "finbert_demeaned",
                    "finbert_polarity_raw", "opinion_mass"]
    st.dataframe(
        filtered[display_cols].sort_values("date", ascending=False).round(4),
        use_container_width=True
    )

# Footer
st.divider()
st.caption(
    "Data: Bank of England MPC Minutes (public domain) . "
    "Model: ProsusAI/finbert via HuggingFace . "
    "Built with Python, FinBERT, pandas, Streamlit"
)