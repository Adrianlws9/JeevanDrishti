import urllib.request
import re
import json

url = "https://dataful.in/datasets/18514/"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=10) as resp:
    html = resp.read().decode("utf-8", errors="ignore")

# Find json-ld or next data
for match in re.finditer(r'<script type="application/ld\+json">(.*?)</script>', html, re.DOTALL):
    try:
        data = json.loads(match.group(1))
        print("JSON-LD:")
        print(json.dumps(data, indent=2)[:1000])
    except Exception as e:
        print("JSON-LD parse error:", e)

# Search for any S3 or file URLs
urls = re.findall(r'https?://[^\s"\'\\]+\.(?:csv|parquet|xlsx|zip)', html, re.I)
print("Direct file URLs in HTML:", set(urls))
