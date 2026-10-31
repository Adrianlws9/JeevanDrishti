import urllib.request
import re

url = "https://dataful.in/datasets/18514/"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
        print("Page length:", len(html))
        # Look for any download, file, api or csv links
        links = re.findall(r"href=[\"']([^\"']*(?:csv|download|api|file|xlsx|parquet)[^\"']*)[\"']", html, re.I)
        print("Matching hrefs:", set(links))
        # Search for buttons or forms
        forms = re.findall(r"<form[^>]*action=[\"']([^\"']*)[\"']", html, re.I)
        print("Forms:", forms)
        # Search for json data in script tags
        scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.DOTALL)
        for s in scripts:
            if "download" in s.lower() or "csv" in s.lower() or "file" in s.lower():
                print("Found relevant script snippet:", s[:300])
except Exception as e:
    print("Error:", e)
