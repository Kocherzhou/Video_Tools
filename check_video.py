import json, subprocess

# Check cache
with open("/mnt/c/Users/tanga/video-subtitle/uploads/6916f3c5953b4f69b5a9ffebb53b6653_cache.json", encoding="utf-8") as f:
    cache = json.load(f)
print("Cached duration:", cache.get("duration"))

# Re-check actual video duration
r = subprocess.run(
    ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
     "/mnt/c/Users/tanga/video-subtitle/uploads/6916f3c5953b4f69b5a9ffebb53b6653.mp4"],
    capture_output=True, text=True
)
info = json.loads(r.stdout)
print("Actual duration:", info["format"]["duration"])
print("File size:", info["format"].get("size"), "bytes")
