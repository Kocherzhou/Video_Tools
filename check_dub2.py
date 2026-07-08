import subprocess, json, wave, os

job = "7bba6196b7fc42b9bcb7ead148afc256"
base = f"/mnt/c/Users/tanga/video-subtitle/uploads/{job}"

# Check dubbed.mp4
r = subprocess.run(
    ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", f"{base}_dubbed.mp4"],
    capture_output=True, text=True
)
info = json.loads(r.stdout)
print("=== dubbed.mp4 streams ===")
for s in info.get("streams", []):
    print(f"  {s['codec_type']} / {s.get('codec_name')} / rate={s.get('sample_rate','')} / dur={s.get('duration','?')}")

# Check seg_0000.wav
seg_path = f"{base}_audio/seg_0000.wav"
print(f"\n=== {seg_path} ===")
try:
    with wave.open(seg_path, "rb") as w:
        dur = w.getnframes() / w.getframerate()
        print(f"  WAV OK: {dur:.2f}s, rate={w.getframerate()}, channels={w.getnchannels()}")
except Exception as e:
    print(f"  WAV error: {e}")
    # Try ffprobe on it
    r2 = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", seg_path],
        capture_output=True, text=True
    )
    try:
        info2 = json.loads(r2.stdout)
        fmt = info2.get("format", {})
        print(f"  ffprobe: format={fmt.get('format_name')}, dur={fmt.get('duration')}")
    except Exception:
        print(f"  ffprobe stdout: {r2.stdout[:200]}")
        print(f"  ffprobe stderr: {r2.stderr[:200]}")

# Check merged.wav size
merged = f"{base}_audio/merged.wav"
if os.path.exists(merged):
    print(f"\n=== merged.wav size: {os.path.getsize(merged)} bytes ===")
    r3 = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", merged],
        capture_output=True, text=True
    )
    try:
        info3 = json.loads(r3.stdout)
        fmt3 = info3.get("format", {})
        print(f"  format={fmt3.get('format_name')}, dur={fmt3.get('duration')}")
    except Exception as e:
        print(f"  error: {e}")
