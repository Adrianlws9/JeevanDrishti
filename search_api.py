import urllib.request
import re

url = "https://dataful.in/datasets/18514/"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=10) as resp:
    html = resp.read().decode("utf-8", errors="ignore")

# Look for mentions of download or download URL pattern
snippets = re.findall(r'.{0,50}(?:download|api|dataset_file).{0,50}', html, re.I)
for s in snippets[:15]:
    print("Snippet:", s.strip())
