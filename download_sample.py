import urllib.request
import os

pdf_url = "https://ncdc.mohfw.gov.in/uploads/weekly_outbreaks/2023/week1.pdf"
out_dir = r"C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI\sample_reports"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "week1_2023.pdf")

req = urllib.request.Request(pdf_url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        content = resp.read()
        with open(out_file, "wb") as f:
            f.write(content)
        print(f"Downloaded successfully! Size: {len(content)} bytes")
except Exception as e:
    print("Download error:", e)
