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

    PATH = "data/raw/implied_inflation"

    df1 = pd.read_excel(f"{PATH}/glc_inflation_monthly_1979_2015.xlsx", sheet_name=4, skiprows=3, header=0, index_col=0)
    df2 = pd.read_excel(f"{PATH}/glc_inflation_monthly_2016_2024.xlsx", sheet_name=4, skiprows=3, header=0, index_col=0)
    df3 = pd.read_excel(f"{PATH}/glc_inflation_monthly_2025_present.xlsx", sheet_name=4, skiprows=3, header=0,
                        index_col=0)

    breakeven = pd.concat([df1, df2, df3])
    breakeven.index = pd.to_datetime(breakeven.index, errors="coerce")
    breakeven = breakeven[breakeven.index.notna()]
    breakeven = breakeven.sort_index()
    breakeven = breakeven[~breakeven.index.duplicated(keep="first")]

    breakeven_5yr = breakeven[5.0]["2015":"2026"].rename("breakeven_5yr")

    return scores, bank_rate, breakeven_5yr

scores_df, bank_rate, breakeven_5yr = load_data()

# Header
st.title("Bank of England MPC Sentiment Tracker")
st.markdown(
    "Tracks sentiment in **Bank of England MPC minutes** using **FinBERT** "
    "(a BERT model fine-tuned on financial text), set against the **Bank Rate** and "
    "**5-year market-implied inflation expectations**. Sentiment is demeaned against a "
    "12-meeting rolling window so deviations reflect shifts in tone, not absolute level."
)

with st.expander("About this dashboard"):
    st.markdown("""
    **What this shows.** Sentiment scores from the Bank of England's Monetary Policy 
    Committee minutes, scored sentence-by-sentence with FinBERT and aggregated per meeting. 
    The score is *demeaned* against a 12-meeting rolling baseline, so the y-axis shows 
    deviations from recent norms rather than absolute values.

    **Reading the chart.** When the blue line is above zero, the meeting was more positive 
    than recent meetings; below zero, more negative. Bank Rate is overlaid on the right axis 
    to compare tone with policy action. The bottom panel when toggled shows market-implied 
    inflation expectations from gilt yields.

    **A note on FinBERT.** FinBERT was trained on financial news and filings, not central 
    bank communications. It captures *outlook* sentiment (good news vs. bad news) more than 
    *policy stance* (hawkish vs. dovish). This implies that a hawkish MPC discussing recession risks may score 
    as negative. A future iteration will add a hawkish/dovish dictionary to separate the two channels.
    """)

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

show_breakevens = st.sidebar.checkbox("Show inflation expectations", value=True)

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
if show_breakevens:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.12,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )
else:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

if show_breakevens:
    fig.update_annotations(font_size=12)
top_kwargs = dict(row=1, col=1) if show_breakevens else {}

if metric == "Components":
    fig.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["positive"],
        mode="lines", name="Positive share",
        line=dict(color="seagreen", width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>Positive: %{y:.3f}<extra></extra>"
    ), secondary_y=False, **top_kwargs)
    fig.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["negative"],
        mode="lines", name="Negative share",
        line=dict(color="indianred", width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>Negative: %{y:.3f}<extra></extra>"
    ), secondary_y=False, **top_kwargs)
else:
    if show_raw:
        fig.add_trace(go.Scatter(
            x=filtered["date"], y=filtered["primary"],
            mode="lines", name=f"Unsmoothed {primary_label.lower()}",
            line=dict(color="steelblue", width=0.8), opacity=0.3,
            hovertemplate="<b>%{x|%b %Y}</b><br>Unsmoothed: %{y:.3f}<extra></extra>"
        ), secondary_y=False, **top_kwargs)

    smoothed_label = (
        primary_label if smoothing == 1
        else f"{primary_label} ({smoothing}-meeting avg)"
    )
    fig.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["primary_smooth"],
        mode="lines", name=smoothed_label,
        line=dict(color="steelblue", width=2.5),
        hovertemplate="<b>%{x|%b %Y}</b><br>Sentiment: %{y:.3f}<extra></extra>"
    ), secondary_y=False, **top_kwargs)

if show_uncertainty:
    fig.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["opinion_mass"],
        mode="lines", name="Opinion mass (pos + neg)",
        line=dict(color="orange", width=1.2, dash="dot"),
        opacity=0.6,
        hovertemplate="<b>%{x|%b %Y}</b><br>Opinion mass: %{y:.3f}<extra></extra>"
    ), secondary_y=False, **top_kwargs)

if show_breakevens:
    bk_mask = (breakeven_5yr.index.date >= date_range[0]) & (breakeven_5yr.index.date <= date_range[1])
    filtered_bk = breakeven_5yr[bk_mask]

    fig.add_trace(go.Scatter(
        x=filtered_bk.index, y=filtered_bk.values,
        mode="lines", name="5yr implied inflation (%)",
        line=dict(color="tomato", width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>Break-even: %{y:.2f}%<extra></extra>"
    ), row=2, col=1)

fig.add_trace(go.Scatter(
    x=filtered_rate["date"], y=filtered_rate["bank_rate"],
    mode="lines", name="Bank Rate (%)",
    line=dict(color="white", width=2, dash="dash"),
    hovertemplate="<b>%{x|%b %Y}</b><br>Bank Rate: %{y:.2f}%<extra></extra>"
), secondary_y=True, **top_kwargs)

fig.add_hline(y=0, line_dash="dot", line_color="grey", line_width=0.8, **top_kwargs)

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

fig.update_yaxes(title_text=y_axis_label, range=sent_range, secondary_y=False, **top_kwargs)
fig.update_yaxes(title_text="Bank Rate (%)", range=rate_range, showgrid=False, secondary_y=True, **top_kwargs)

if show_breakevens:
    bk_min, bk_max = filtered_bk.min(), filtered_bk.max()
    bk_pad = max((bk_max - bk_min) * 0.15, 0.3)
    fig.update_yaxes(
        title_text="Inflation (%)",
        range=[max(1.5, bk_min - bk_pad), bk_max + bk_pad],
        dtick=0.5,
        row=2, col=1
    )
# X-axis formatting
xaxis_config = dict(
    tickformat="%b %Y",
    dtick="M12",          # tick every 12 months
    tickangle=-45,
    showgrid=True,
    gridcolor="rgba(128,128,128,0.2)",
    tickfont=dict(size=11),
)

if show_breakevens:
    fig.update_xaxes(**xaxis_config, row=1, col=1, showticklabels=True)
    fig.update_xaxes(**xaxis_config, title_text="", row=2, col=1, showticklabels=True)
else:
    fig.update_xaxes(**xaxis_config)

fig.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
    hovermode="x unified",
    height=850 if show_breakevens else 550,
    template="plotly_dark",
    margin=dict(t=80)
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
    "Data: Bank of England MPC Minutes (public domain) | "
    "Model: ProsusAI/finbert via HuggingFace | "
    "Built with Python, FinBERT, pandas, Streamlit | "
    "Built by Andreas Drymiotis | "
    "[Github](https://github.com/adrymmm) | "
    "[LinkedIn](https://www.linkedin.com/in/andreas-drymiotes-a02293295) |"
)