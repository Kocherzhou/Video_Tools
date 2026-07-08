import json

with open("/mnt/c/Users/tanga/video-subtitle/uploads/6916f3c5953b4f69b5a9ffebb53b6653_subtitles.json", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total subtitles: {len(data)}")
for i, s in enumerate(data[:5]):
    print(f"\n[{i}] {s.get('start',0):.1f}s - {s.get('end',0):.1f}s")
    print(f"  Translation: {s.get('translation','')[:100]}")
