import subprocess, json

path = "/mnt/c/Users/tanga/video-subtitle/uploads/7bba6196b7fc42b9bcb7ead148afc256_dubbed.mp4"
audio_dir = "/mnt/c/Users/tanga/video-subtitle/uploads/7bba6196b7fc42b9bcb7ead148afc256_audio"

# Check dubbed.mp4 streams
r = subprocess.run(
    ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
    capture_output=True, text=True
)
info = json.loads(r.stdout)
print("=== Streams in dubbed.mp4 ===")
for s in info.get("streams", []):
    print(f"  Stream {s['index']}: {s['codec_type']} / {s.get('codec_name')} / {s.get('sample_rate','')}")

# Check merged.wav
import wave
try:
    with wave.open(f"{audio_dir}/merged.wav", "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        dur = frames / rate
        print(f"\n=== merged.wav ===")
        print(f"  Duration: {dur:.2f}s, Frames: {frames}, Rate: {rate}")
except Exception as e:
    print(f"merged.wav error: {e}")

# Check first few seg files
import os
segs = sorted([f for f in os.listdir(audio_dir) if f.startswith("seg_")])
print(f"\n=== seg files ({len(segs)} total) ===")
for seg in segs[:3]:
    spath = f"{audio_dir}/{seg}"
    try:
        with wave.open(spath, "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
            print(f"  {seg}: {dur:.2f}s")
    except Exception as e:
        print(f"  {seg}: error {e}")
