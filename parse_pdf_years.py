import urllib.request
import re

url = "https://ncdc.mohfw.gov.in/includes/WeeklyOutbreaks.php"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=10) as resp:
    html = resp.read().decode("utf-8", errors="ignore")

pdf_links = re.findall(r'href=[\'"]([^\'"]*weekly_outbreaks[^\'"]*\.pdf)[\'"]', html, re.I)

y2023 = [p for p in pdf_links if '2023' in p]
y2024 = [p for p in pdf_links if '2024' in p]
y2025 = [p for p in pdf_links if '2025' in p]

print(f"Total 2023 reports: {len(y2023)}")
print(f"Total 2024 reports: {len(y2024)}")
print(f"Total 2025 reports: {len(y2025)}")
print("Sample 2023:", y2023[:3])
print("Sample 2024:", y2024[:3])
print("Sample 2025:", y2025[:3])
