r"""
Whisper GPU diagnostic. Run from your project dir:
  python test_whisper.py uploads\YOUR_VIDEO.mp4

This actually runs inference (not just loads model) so we can see
whether transcription is really hitting the GPU.

In a SECOND PowerShell window, run this to watch GPU live:
  nvidia-smi -l 1
"""
import sys, time, os
from pathlib import Path

# Make sure pip-installed CUDA wheels are findable on Windows.
# Windows wheel layout: nvidia/<pkg>/bin/*.dll (NOT nvidia/<pkg>/lib/ — that's Linux)
# IMPORTANT: ctranslate2's compiled binary uses legacy LoadLibrary which doesn't
# respect os.add_dll_directory(). We MUST also prepend to PATH.
nvidia_dirs = []
if os.name == "nt":
    for pkg_name in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            pkg = __import__(pkg_name, fromlist=[""])
            pkg_dir = os.path.dirname(pkg.__file__)
            for subdir in ("bin", "lib"):
                candidate = os.path.join(pkg_dir, subdir)
                if os.path.isdir(candidate):
                    os.add_dll_directory(candidate)
                    nvidia_dirs.append(candidate)
                    print(f"[+] added {pkg_name} DLL dir: {candidate}")
                    dlls = [f for f in os.listdir(candidate) if f.endswith(".dll")][:3]
                    if dlls:
                        print(f"    sample DLLs: {dlls}")
                    break
            else:
                print(f"[!] {pkg_name} found at {pkg_dir} but no bin/ or lib/ subdir")
        except ImportError as e:
            print(f"[!] {pkg_name} not installed: {e}")
        except Exception as e:
            print(f"[!] {pkg_name} setup error: {e}")

    # Critical: prepend nvidia bin dirs to PATH so ctranslate2's LoadLibrary finds them
    if nvidia_dirs:
        os.environ["PATH"] = os.pathsep.join(nvidia_dirs) + os.pathsep + os.environ.get("PATH", "")
        print(f"[+] prepended {len(nvidia_dirs)} dirs to PATH")

if len(sys.argv) < 2:
    print("usage: python test_whisper.py <video_path>")
    sys.exit(1)

video = sys.argv[1]
if not Path(video).exists():
    print(f"file not found: {video}")
    sys.exit(1)

from faster_whisper import WhisperModel

# Start small to verify GPU works before trying the big model
model_size = sys.argv[2] if len(sys.argv) > 2 else "tiny"
print(f"\n=== Test 1: load {model_size} on CUDA ===")
t = time.time()
m = WhisperModel(model_size, device="cuda", compute_type="float16")
print(f"loaded in {time.time()-t:.1f}s")

print(f"\n=== Test 2: actually transcribe ===")
t = time.time()
segments, info = m.transcribe(video, language="en", vad_filter=True)
print(f"video duration: {info.duration:.1f}s, language={info.language} ({info.language_probability:.2f})")

n = 0
first_seg_time = None
for seg in segments:
    n += 1
    if first_seg_time is None:
        first_seg_time = time.time() - t
        print(f"first segment at wall-clock {first_seg_time:.1f}s: [{seg.start:.1f}-{seg.end:.1f}] {seg.text[:80]}")
    if n <= 5 or n % 10 == 0:
        elapsed = time.time() - t
        progress = seg.end / info.duration * 100 if info.duration else 0
        print(f"  seg {n:3d} | wall {elapsed:5.1f}s | video pos {seg.end:6.1f}s / {info.duration:.0f}s ({progress:4.1f}%)")

total = time.time() - t
speed = info.duration / total if total > 0 else 0
print(f"\n=== Result ===")
print(f"total: {total:.1f}s for {info.duration:.1f}s video → {speed:.1f}x realtime, {n} segments")
print(f"on 3080+CUDA, tiny should hit 100x+. Below 10x = running on CPU.")
