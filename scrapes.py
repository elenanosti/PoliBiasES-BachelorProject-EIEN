import os
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

# Main execution
info = pd.read_csv("clean_ALL_complete_congreso_links_motion_titel_category_subcat_motionid.csv")
initiative_urls = info["URL"].dropna().unique().tolist()

all_records = []
chunk_size = 200
chunk_count = 0

for i, initiative_url in enumerate(tqdm(initiative_urls, desc="Initiatives")):
    json_urls = get_json_urls_from_initiative(initiative_url)
    for json_url in tqdm(json_urls, desc="  JSONs", leave=False):
        records = extract_votes_from_json(json_url)
        for rec in records:
            rec["initiative_url"] = initiative_url
        all_records.extend(records)

        if len(all_records) >= chunk_size:
            chunk_filename = f"votes_chunk_{chunk_count}.csv"
            pd.DataFrame(all_records).to_csv(chunk_filename, index=False, encoding="utf-8")

            # Git operations
            os.system(f"git pull origin main")
            os.system(f"git add {chunk_filename}")
            os.system(f"git commit -m 'Auto upload: chunk {chunk_count}'")
            os.system(f"git push origin main")
            os.remove(chunk_filename)

            all_records.clear()
            chunk_count += 1

# Final flush
if all_records:
    final_chunk = f"votes_chunk_{chunk_count}.csv"
    pd.DataFrame(all_records).to_csv(final_chunk, index=False, encoding="utf-8")
    os.system(f"git pull origin main")
    os.system(f"git add {final_chunk}")
    os.system(f"git commit -m 'Final chunk {chunk_count}'")
    os.system(f"git push origin main")
    os.remove(final_chunk)

print("✅ Scraping and upload completed.")