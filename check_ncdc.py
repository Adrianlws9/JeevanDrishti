import urllib.request
import re

url = "https://ncdc.mohfw.gov.in/"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
        print("NCDC Main Page status:", resp.status)
        matches = re.findall(r'href=[\'"]([^\'"]*outbreak[^\'"]*)[\'"]', html, re.I)
        print("Outbreak matches:", matches[:10])
except Exception as e:
    print("NCDC Main Page Error:", e)
