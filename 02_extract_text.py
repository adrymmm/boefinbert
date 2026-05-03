import pdfplumber
import pandas as pd
from pathlib import Path

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def clean_text(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        # Skip page numbers, headers, short lines
        if len(line) < 40:
            continue
        # Skip common boilerplate
        if any(skip in line for skip in [
            "Bank of England", "Monetary Policy Committee",
            "All rights reserved", "www.bankofengland"
        ]):
            continue
        cleaned.append(line)
    return " ".join(cleaned)

# Process all PDFs
records = []
pdf_dir = Path("minutes")

for pdf_path in sorted(pdf_dir.glob("*.pdf")):
    # Extract date from filename
    parts = pdf_path.stem.split("-")
    year = int(parts[0])
    month = parts[1]

    raw_text = extract_text(pdf_path)
    clean = clean_text(raw_text)

    records.append({
        "year": year,
        "month": month,
        "date_str": f"{month} {year}",
        "text": clean
    })
    print(f"✓ Extracted: {pdf_path.name} ({len(clean)} chars)")

# Save to CSV
df = pd.DataFrame(records)
df.to_csv("minutes_text.csv", index=False)
print(f"\nDone. {len(df)} documents saved to minutes_text.csv")