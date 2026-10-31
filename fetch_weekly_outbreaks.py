import urllib.request
import re

url = "https://ncdc.mohfw.gov.in/includes/WeeklyOutbreaks.php"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
        print("WeeklyOutbreaks page loaded, length:", len(html))
        # Look for links to pdf or archives
        pdf_links = re.findall(r'href=[\'"]([^\'"]*\.pdf)[\'"]', html, re.I)
        print("PDF links count:", len(pdf_links))
        for p in pdf_links[:15]:
            print("  PDF:", p)
        # Look for years or sections
        years = re.findall(r'(202[3-5])', html)
        print("Recent years mentioned:", set(years))
except Exception as e:
    print("Error:", e)
