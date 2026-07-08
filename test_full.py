"""Full integration test: subtitle + dubbing + PDF slide dub"""
import requests, time, json, os, sys
from pathlib import Path

BASE = "http://localhost:5000"
VIDEO = r"F:\downloads\The_AI-Native_Reconstruct.mp4"
PDF   = r"F:\downloads\2026_Strategic_War_Assessment.pdf"
VOICE = "edge_yuyang"
UPLOADS = Path(r"C:\Users\tanga\video-subtitle\uploads")

def log(msg, color=""):
    colors = {"green":"\033[92m","red":"\033[91m","cyan":"\033[96m","yellow":"\033[93m","":""}
    reset = "\033[0m" if color else ""
    t = time.strftime("%H:%M:%S")
    print(f"{colors.get(color,'')}{t} {msg}{reset}", flush=True)

def poll_job(job_id, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE}/api/job_info/{job_id}", timeout=5).json()
            status = r.get("status", "?")
            progress = r.get("progress", 0)
            message = r.get("message", "")
            log(f"[{progress}%] {status} — {message}")
            if status == "completed":
                return r
            if status == "error":
                log(f"ERROR: {message}", "red")
                return None
        except Exception as e:
            log(f"waiting... ({e})")
        time.sleep(5)
    log("Timeout!", "red")
    return None

# ── TEST 1: Video subtitle generation ─────────────────────────
log("=== TEST 1: Subtitle Generation ===", "cyan")
with open(VIDEO, "rb") as f:
    resp = requests.post(f"{BASE}/api/upload", files={"video": (Path(VIDEO).name, f, "video/mp4")})
job_id = resp.json()["job_id"]
log(f"Job ID: {job_id}")

result = poll_job(job_id, 300)
if not result:
    log("Subtitle generation FAILED", "red")
    sys.exit(1)

subs = result.get("subtitles", [])
log(f"Generated {len(subs)} subtitles", "green")
for s in subs[:3]:
    t = s.get("translation","")[:50]
    log(f"  {s['start']:.1f}s-{s['end']:.1f}s: {t}")

# Check all subtitles >= 6 seconds
short = [s for s in subs if (s['end']-s['start']) < 6]
if short:
    log(f"WARNING: {len(short)} subtitles < 6s", "yellow")
else:
    log("All subtitles >= 6s ✓", "green")

# Check char counts
violations = []
for s in subs:
    slot = s['end'] - s['start']
    max_chars = slot * 4.5
    actual = len(s.get('translation',''))
    if actual > max_chars:
        violations.append(f"{s['start']:.1f}s: {actual}chars > {max_chars:.0f}max")
if violations:
    log(f"Char limit violations: {len(violations)}", "yellow")
    for v in violations[:3]:
        log(f"  {v}")
else:
    log("All char counts within 4.5x limit ✓", "green")

# ── TEST 2: Dubbing ────────────────────────────────────────────
log(f"\n=== TEST 2: Dubbing ({VOICE}) ===", "cyan")
requests.post(f"{BASE}/api/dub/{job_id}", json={"voice": VOICE})

deadline = time.time() + 300
while time.time() < deadline:
    time.sleep(5)
    dubbed = UPLOADS / f"{job_id}_dubbed.mp4"
    if dubbed.exists():
        mb = dubbed.stat().st_size / 1024 / 1024
        log(f"dubbed.mp4: {mb:.1f} MB", "green")
        break
    segs = list((UPLOADS / f"{job_id}_audio").glob("seg_*.wav")) if (UPLOADS / f"{job_id}_audio").exists() else []
    log(f"TTS segments: {len(segs)}/{len(subs)}")
else:
    log("Dubbing TIMEOUT", "red")

# ── TEST 3: PDF slide dubbing ──────────────────────────────────
log(f"\n=== TEST 3: PDF Slide Dub ===", "cyan")
if not Path(PDF).exists():
    log(f"PDF not found: {PDF}", "red")
else:
    with open(PDF, "rb") as f:
        resp = requests.post(
            f"{BASE}/api/slide_dub_start/{job_id}",
            files={"file": (Path(PDF).name, f, "application/pdf")},
            data={"voice": VOICE}
        )
    task_id = resp.json().get("task_id")
    log(f"Slide task: {task_id}")

    deadline = time.time() + 600
    while time.time() < deadline:
        time.sleep(8)
        try:
            # Check slide task status via internal dict not exposed - check output file
            slide_dir = UPLOADS / f"{task_id}_slides"
            narrations = slide_dir / "narrations.json"
            output_mp3 = slide_dir / "output.mp3"
            output_mp4 = slide_dir / "output.mp4"

            pages = list(slide_dir.glob("page-*.jpg")) if slide_dir.exists() else []
            done_segs = list(slide_dir.glob("page_*_raw.wav")) if slide_dir.exists() else []
            narr_count = len(json.load(open(narrations, encoding="utf-8"))) if narrations.exists() else 0

            log(f"Pages: {len(pages)}, Narrations: {narr_count}, Audio segs: {len(done_segs)}")

            if output_mp4.exists():
                mb = output_mp4.stat().st_size / 1024 / 1024
                log(f"Slide output.mp4: {mb:.1f} MB ✓", "green")
                break
        except Exception as e:
            log(f"Checking... ({e})")
    else:
        log("Slide dub TIMEOUT", "red")

log("\n=== All Tests Done ===", "yellow")
log(f"Browser job ID: {job_id}")
