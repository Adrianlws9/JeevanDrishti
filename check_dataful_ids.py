import urllib.request
import re

datasets = [15409, 15725, 15726, 15728, 15733, 18514, 19434, 21275, 21276, 21277, 21278, 21279]

for did in datasets:
    url = f"https://dataful.in/datasets/{did}/"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
            title = title_match.group(1) if title_match else "No title"
            download_links = re.findall(r"href=[\"'](/download/[^\"']+)[\"']", html)
            print(f"ID {did}: {title.strip()} | Downloads: {download_links[:3]}")
    except Exception as e:
        print(f"ID {did} Error: {e}")
