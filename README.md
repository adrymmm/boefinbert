# BoE MPC Sentiment Analysis
## **🔗 [Live dashboard](https://boe-mpc-sentiment.streamlit.app/)**

NLP sentiment analysis tool that runs **FinBERT** (a BERT model fine-tuned on financial texts)
on Bank of England Monetary Policy Committee (MPC) minutes,
analysing the tone of UK monetary policy communications between 2015-2026.

![Dashboard Screenshpt](assets/web_chart.png)

---

## Motivation
MPC communications are themselves policy instruments, updating expectations about growth, inflation and market pricing.
The project aims to make the shift in tone withing those meetings quantifiable and explorable; it is plotted with the
Bank Rate series to make the relationship explicit.

---

## Features
 
- **Net sentiment (demeaned)** — FinBERT scores aggregated per meeting, expressed as deviation from a 12-meeting rolling baseline to remove long-run drift
- **Polarity (raw)** — direct positive minus negative score per meeting, without demeaning; reflects the absolute level of
positivity/negativity
- **Components view** — positive, negative, and neutral sentence shares plotted separately, showing the composition of tone rather than a single index
- **Base rate overlay** — sentiment plotted against the official Bank Rate to visualise the relationship between language and policy
- **Smoothing controls** — rolling average window to surface trends through meeting-to-meeting noise
- **Date range filter** — zoom into specific episodes (GFC, Covid shock, 2021–23 inflation cycle)
---

## Results
### Key findings

**2022 sentiment trough** -- net sentiment hit its most negative reading in the sample during the 2022 inflation surge, coinciding with the most aggressive rate-hiking cycle in the BoE's modern history. The language turned sharply negative as CPI peaked above 11%.

**Sentiment leads the narrative** -- the demeaned sentiment index began deteriorating in late 2021, several meetings before the first rate rise in December 2021, consistent with the MPC signalling concern ahead of action.

**Post-peak recovery** -- sentiment recovered through 2023–24 as inflation fell back toward target, tracking the slowdown in rate rises and eventual cuts.

### Limitations

- FinBERT was fine-tuned on financial news and filings, not central bank communications specifically (domain mismatch)
- Chunking at 400 characters treats all parts of the minutes equally; the opening summary and voting section likely carry different informational weight
- Demeaning against a 12-meeting rolling window means the earliest observations have less stable baselines

---

## Pipeline
 
```
MPC PDFs (BoE website)
        │
        v
  requests (download PDFs)
        |
        v
  pdfplumber (extract text + clean output)
        │
        v
  FinBERT (ProsusAI/finbert via HuggingFace)
  + positive / negative / neutral scores per sentence
        │
        v
  pandas (aggregate scores -> compute meeting-level sentiment index)
        │
        v
  sentiment_scores_full.csv
        │
        v
  Streamlit dashboard (deployed on Streamlit Cloud)
  + Exploration and illustration
```

Model inference runs locally in [03_sentiment_analysis.ipynb](notebooks/03_sentiment_analysis.ipynb) and creates a CSV of aggregated scores.
Streamlit reads from the CSV and creates an interactive dashboard.

---
## Repo Structure

```
boefinbert/
├── app.py                        # Streamlit app entry point
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── 01_download_minutes.py    # Fetch MPC PDFs from BoE website
│   └── 02_extract_text.py        # PDF extraction + FinBERT scoring
├── notebooks/
│   └── 03_sentiment_analysis.ipynb
├── data/
│   ├── raw/
│   │   ├── minutes/              # MPC PDF files
│   │   ├── bank_rate.csv
│   │   └── minutes_text.csv
│   └── processed/
│       └── sentiment_scores_full.csv
└── assets/
    └── web_chart.png
```
---

## Running Locally
 
```bash
git clone https://github.com/adrymmm/boefinbert.git
cd boefinbert
pip install -r requirements.txt
streamlit run app.py
```
---

## Potential Extensions
- Granger causality test: test whether MPC sentiment lead rate changes?
- Loughran-McDonald dictionary baseline comparison

---