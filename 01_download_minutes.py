import requests
import time
from pathlib import Path

Path("data/minutes").mkdir(exist_ok=True)

base = "https://www.bankofengland.co.uk/-/media/boe/files/monetary-policy-summary-and-minutes"

def download_minutes(month, year):
    base = "https://www.bankofengland.co.uk/-/media/boe/files/monetary-policy-summary-and-minutes"

    urls_to_try = [
        f"{base}/{year}/monetary-policy-summary-and-minutes-{month}-{year}.pdf",
        f"{base}/{year}/minutes-{month}-{year}.pdf",
        f"{base}/{year}/{month}-{year}.pdf"
    ]

    for url in urls_to_try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            filename = f"data/minutes/{year}-{month}.pdf"
            with open(filename, "wb") as f:
                f.write(r.content)
            print(f"✓ {year} {month}")
            return True

    print(f"✗ Failed: {month} {year}")
    return False

meetings = [
    # 2015 (meetings moved to 8x/year from August 2015)
    ("january", 2015), ("february", 2015), ("march", 2015), ("may", 2015), ("june", 2015),
    ("august", 2015), ("september", 2015), ("november", 2015), ("december", 2015),
    # 2016
    ("february", 2016), ("march", 2016), ("may", 2016), ("june", 2016),
    ("august", 2016), ("september", 2016), ("november", 2016), ("december", 2016),
    # 2017 onwards - same as before
    ("february", 2017), ("march", 2017), ("may", 2017), ("june", 2017),
    ("august", 2017), ("september", 2017), ("november", 2017), ("december", 2017),
    ("february", 2018), ("march", 2018), ("may", 2018), ("june", 2018),
    ("august", 2018), ("september", 2018), ("november", 2018), ("december", 2018),
    ("january", 2019), ("march", 2019), ("may", 2019), ("june", 2019),
    ("august", 2019), ("september", 2019), ("november", 2019), ("december", 2019),
    ("january", 2020), ("march", 2020), ("may", 2020), ("june", 2020),
    ("august", 2020), ("september", 2020), ("november", 2020), ("december", 2020),
    ("february", 2021), ("march", 2021), ("may", 2021), ("june", 2021),
    ("august", 2021), ("september", 2021), ("november", 2021), ("december", 2021),
    ("february", 2022), ("march", 2022), ("may", 2022), ("june", 2022),
    ("august", 2022), ("september", 2022), ("november", 2022), ("december", 2022),
    ("february", 2023), ("march", 2023), ("may", 2023), ("june", 2023),
    ("august", 2023), ("september", 2023), ("november", 2023), ("december", 2023),
    ("february", 2024), ("march", 2024), ("may", 2024), ("june", 2024),
    ("august", 2024), ("september", 2024), ("november", 2024), ("december", 2024),
    ("february", 2025), ("march", 2025), ("may", 2025), ("june", 2025),
    ("august", 2025), ("september", 2025), ("november", 2025), ("december", 2025),
]

failed = []
for month, year in meetings:
    if not download_minutes(month, year):
        failed.append((month, year))
    time.sleep(1)

if failed:
    print(f"\nFailed: {failed}")
else:
    print("\nAll done.")