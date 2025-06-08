
with open("congreso_scraper.py", "w") as f:
    f.write('''\
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
from tqdm import tqdm

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}
BASE_URL = "https://www.congreso.es"

def get_json_urls_from_initiative(initiative_url):
    try:
        res = requests.get(initiative_url, headers=HEADERS)
        soup = BeautifulSoup(res.content, "html.parser")
        json_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.json')]
        return [urljoin(BASE_URL, link) for link in json_links]
    except Exception as e:
        print(f"❌ Failed to fetch JSON links from {initiative_url}: {e}")
        return []

def extract_votes_from_json(json_url):
    try:
        res = requests.get(json_url, headers=HEADERS)
        data = res.json()

        info = data.get("informacion", {})
        totals = data.get("totales", {})
        votes = data.get("votaciones", [])

        records = []
        if not votes:
            records.append({
                "date": info.get("fecha"),
                "title": info.get("titulo"),
                "motion_text": info.get("textoExpediente"),
                "deputy": None,
                "party": None,
                "vote": None,
                "afavor": totals.get("afavor"),
                "enContra": totals.get("enContra"),
                "abstenciones": totals.get("abstenciones"),
                "noVotan": totals.get("noVotan")
            })
        else:
            for vote in votes:
                records.append({
                    "date": info.get("fecha"),
                    "title": info.get("titulo"),
                    "motion_text": info.get("textoExpediente"),
                    "deputy": vote.get("diputado"),
                    "party": vote.get("grupo"),
                    "vote": vote.get("voto"),
                    "afavor": totals.get("afavor"),
                    "enContra": totals.get("enContra"),
                    "abstenciones": totals.get("abstenciones"),
                    "noVotan": totals.get("noVotan")
                })
        return records
    except Exception as e:
        print(f"❌ Failed to process {json_url}: {e}")
        return []
''')
    
import pandas as pd
from congreso_scraper import get_json_urls_from_initiative, extract_votes_from_json
from tqdm import tqdm

# Load the CSV with all initiative URLs
info = pd.read_csv("clean_ALL_complete_congreso_links_motion_titel_category_subcat_motionid.csv")
initiative_urls = info["URL"].dropna().unique().tolist()

# LIMIT URLS
#initiative_urls = initiative_urls[:20] #Test only first 20

all_records = []

for initiative_url in tqdm(initiative_urls, desc="Initiatives"):
    json_urls = get_json_urls_from_initiative(initiative_url)
    for json_url in tqdm(json_urls, desc="  JSONs", leave=False):
        records = extract_votes_from_json(json_url)
        for rec in records:
            rec["initiative_url"] = initiative_url  # Add URL for merging
        all_records.extend(records)

# Save the votes with URL for merging
df = pd.DataFrame(all_records)
df.to_csv("ALL_Votes_legislature_with_url.csv", index=False, encoding="utf-8")
print("✅ All votes with URLs saved as 'TestVotes_legislatureXV_with_url.csv'")

