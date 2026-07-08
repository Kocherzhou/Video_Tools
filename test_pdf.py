"""Test PDF slide dubbing only"""
import requests, time, json, os
from pathlib import Path

BASE = "http://localhost:5000"
PDF  = r"F:\downloads\2026_Strategic_War_Assessment.pdf"
VOICE = "edge_yuyang"
UPLOADS = Path(r"C:\Users\tanga\video-subtitle\uploads")

# Need an existing job_id for the slide dub to reference
# Use a previously generated one or create a dummy
# Let's use the latest job we generated
existing_jobs = sorted(UPLOADS.glob("*_subtitles.json"), key=os.path.getmtime, reverse=True)
if existing_jobs:
    job_id = existing_jobs[0].stem.replace("_subtitles","")
    print(f"Using existing job: {job_id}")
else:
    print("No existing jobs found, creating dummy")
    job_id = "dummy"

print(f"Uploading PDF: {PDF}")
with open(PDF, "rb") as f:
    resp = requests.post(
        f"{BASE}/api/slide_dub_start/{job_id}",
        files={"file": (Path(PDF).name, f, "application/pdf")},
        data={"voice": VOICE}
    )
task_id = resp.json().get("task_id")
print(f"Task: {task_id}")

deadline = time.time() + 600
while time.time() < deadline:
    time.sleep(10)
    slide_dir = UPLOADS / f"{task_id}_slides"
    pages = list(slide_dir.glob("page-*.jpg")) if slide_dir.exists() else []
    narrations = slide_dir / "narrations.json"
    narr_count = len(json.load(open(narrations, encoding="utf-8"))) if narrations.exists() else 0
    done_segs = list(slide_dir.glob("page_*_raw.wav")) if slide_dir.exists() else []
    output_mp3 = slide_dir / "output.mp3"
    output_mp4 = slide_dir / "output.mp4"
    
    t = time.strftime("%H:%M:%S")
    print(f"{t} Pages: {len(pages)}, Narrations: {narr_count}, Audio: {len(done_segs)}", flush=True)
    
    if output_mp4.exists():
        mb = output_mp4.stat().st_size / 1024 / 1024
        print(f"DONE! output.mp4: {mb:.1f} MB")
        break
    elif output_mp3.exists() and not output_mp4.exists():
        mb_mp3 = output_mp3.stat().st_size / 1024 / 1024
        print(f"MP3 done ({mb_mp3:.1f} MB), waiting for MP4...")

print("Test complete.")
