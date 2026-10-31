import urllib.request
import re

url = "https://dataful.in/datasets/?q=IDSP"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
        matches = set(re.findall(r"/datasets/\d+/[^\"'\s>]*", html))
        print("Found links:")
        for m in sorted(matches):
            print(m)
except Exception as e:
    print("Error:", e)
