"""
Video Subtitle Generator - Flask Backend
"""
import os, sys, json, time, uuid, threading, subprocess, tempfile, re, math, hashlib, hmac, unicodedata
from pathlib import Path
from flask import Flask, request, jsonify, Response, send_file, stream_with_context, render_template, make_response, redirect
from mtv_processor import (
    process_mtv_job,
    ALLOWED_MUSIC_EXTS,
    ALLOWED_VIDEO_EXTS as MTV_VIDEO_EXTS,
    ALLOWED_IMAGE_EXTS as MTV_IMAGE_EXTS,
    ALLOWED_PDF_EXTS,
    librosa_available,
    pymupdf_available,
    pdf_support_available,
)
from subtitle_asr import (transcribe_to_subtitles, transcribe_hybrid, SubtitleError,
                          DEFAULT_MODEL as TRANSCRIBE_MODEL,
                          normalize_punct, strip_edge_punct, to_simplified)
# 中文原文模式：每行目标字数（×2 行为单条上限）。约 16 字/行符合中文字幕惯例，
# 保证预览/烧录每屏不超过 2 行。可经 TRANSCRIBE_LINE_CHARS 覆盖。
TRANSCRIBE_LINE_CHARS = int(os.environ.get("TRANSCRIBE_LINE_CHARS", "16"))
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
except ImportError:
    pass

# Ensure local pip bins on PATH
_local_bin = os.path.expanduser("~/.local/bin")
if _local_bin not in os.environ.get("PATH", ""):
    _sep = ";" if os.name == "nt" else ":"
    os.environ["PATH"] = _local_bin + _sep + os.environ.get("PATH", "")

app = Flask(__name__)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
FISH_API_KEY   = os.environ.get("FISH_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
AUTH_TOKEN     = os.environ.get("AUTH_TOKEN", "").strip()

# ── Gemini API key pool (multi-key rotation) ────────────────────────
# Rotating across several keys turns a 429 (free-tier rate limit) into an
# instant switch to a fresh key instead of a 15-60s sleep — the whole point of
# keeping a pool. Keys (AIza...) are extracted from GEMINI_KEYS_FILE if present,
# else we fall back to the single GOOGLE_API_KEY. The file is gitignored.
GEMINI_KEYS_FILE = os.environ.get("GEMINI_KEYS_FILE", str(Path(__file__).parent / "gemini_keys.txt"))

def _load_gemini_keys():
    # 扫描多个 key 文件:项目本地 GEMINI_KEYS_FILE + 复用新闻脚本的
    # ~/news-brief/gemini_keys.txt(两机共用同一批 key),去重后再补 GOOGLE_API_KEY。
    keys = []
    for src in (GEMINI_KEYS_FILE, os.path.expanduser("~/news-brief/gemini_keys.txt")):
        try:
            p = Path(src)
            if p.exists():
                # latin-1 preserves every byte, so ASCII keys survive whatever the
                # comment encoding is (this file's Chinese comments are GBK). Match
                # both the classic AIza... format and the newer AQ.* format.
                text = p.read_text(encoding="latin-1")
                keys += re.findall(
                    r"(?:AIza[0-9A-Za-z_\-]{35}|AQ\.[0-9A-Za-z_\-]{30,80})", text)
        except Exception as e:
            print(f"[gemini-keys] 读取 {src} 失败: {e}")
    if GOOGLE_API_KEY:
        keys.append(GOOGLE_API_KEY)
    return list(dict.fromkeys(keys))  # 去重,保序

GEMINI_KEYS = _load_gemini_keys()
# If only the keys file is provided (no GOOGLE_API_KEY env), adopt the first key
# so all the `if not GOOGLE_API_KEY` guards and status reporting still work.
if not GOOGLE_API_KEY and GEMINI_KEYS:
    GOOGLE_API_KEY = GEMINI_KEYS[0]
_gemini_key_idx = 0
_gemini_key_lock = threading.Lock()
# Transient-failure cooldown: a key that 429s / 503s / momentarily reports
# invalid rests for GEMINI_KEY_COOLDOWN seconds, then rejoins the pool. We do
# NOT permanently disable keys — they're treated as valid-but-occasionally-busy.
_key_cooldown = {}  # key -> epoch seconds to skip until
GEMINI_KEY_COOLDOWN = float(os.environ.get("GEMINI_KEY_COOLDOWN", "60"))

# ── Whisper config (local transcription via faster-whisper) ─────────
# When enabled and the package is installed, transcribe locally for precise
# word-level timestamps, then send only text to Gemini for translation.
# Fixes A/V sync issues that cheaper Gemini models exhibit with timestamps.
WHISPER_ENABLED    = os.environ.get("WHISPER_ENABLED", "1") != "0"
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "large-v3-turbo")
WHISPER_DEVICE     = os.environ.get("WHISPER_DEVICE", "cuda")  # "cuda" or "cpu"
# int8 default (was float16): on this RTX 3080 the GDDR6X / TIM runs hot and the
# card has dropped off the bus under sustained GPU load (see ALLOW_NVENC=0 in
# .env). int8 cuts memory-bandwidth + compute pressure → cooler, fewer power
# spikes, with negligible quality loss on large-v3-turbo. Set float16 to revert.
WHISPER_COMPUTE    = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
# Idle unload: free GPU VRAM (~2.3GB on RTX 3080) when Whisper hasn't been
# used for this many seconds, so video-generation workloads (Wan2GP /
# InfiniteTalk) on the same GPU get the memory back. The model reloads from
# the local HF cache in a few seconds on the next request. 0 disables.
WHISPER_IDLE_UNLOAD_SEC = int(os.environ.get("WHISPER_IDLE_UNLOAD_SEC", "600"))

# Gemini 3.x thinking level — used by gemini-3.5-flash and other Gemini 3 models.
# Translation is closer to "editing" than "complex reasoning", so LOW gives
# the quality benefit of the 3.5 family without paying the MEDIUM/HIGH latency
# tax. Options: minimal | low | medium | high. Default low.
GEMINI_THINKING_LEVEL = os.environ.get("GEMINI_THINKING_LEVEL", "high")

# Whisper transcription tuning
# - condition_on_previous_text: helps Whisper produce more coherent segment
#   boundaries (sentences less likely to be cut in half). Was disabled earlier
#   to fight end-of-video hallucinations, but with the duration filter
#   (info.duration clamp) catching those, we can re-enable for better cuts.
# - vad_min_silence_ms: shorter values segment at micro-pauses (breath),
#   producing more fragments. 500ms keeps mid-sentence breaths together.
WHISPER_CONDITION_ON_PREV = os.environ.get("WHISPER_CONDITION_ON_PREV", "1") != "0"
WHISPER_VAD_MIN_SILENCE_MS = int(os.environ.get("WHISPER_VAD_MIN_SILENCE_MS", "500"))
# Hallucination filtering — strict thresholds drop noisy / low-confidence
# transcriptions that often appear at video endings (music, silence,
# background noise misread as plausible English).
WHISPER_NO_SPEECH_THRESHOLD = float(os.environ.get("WHISPER_NO_SPEECH_THRESHOLD", "0.7"))
WHISPER_LOG_PROB_THRESHOLD = float(os.environ.get("WHISPER_LOG_PROB_THRESHOLD", "-0.8"))

# ── whisper.cpp backend (CPU; for ARM64 / Surface Pro where CUDA is absent) ──
# faster-whisper depends on ctranslate2, which has no Windows-ARM64 wheel, so on
# the Snapdragon X / ARM64 box we shell out to a prebuilt whisper.cpp CLI (x64,
# running under Windows-on-ARM emulation). It produces the exact same
# {start,end,text} segments, so the rest of the pipeline is untouched, and the
# early dispatch in transcribe_with_whisper() also bypasses the GPU thermal guard
# (no CUDA on ARM64).
#   WHISPER_BACKEND = auto | faster-whisper | whispercpp
#   auto → use faster-whisper if importable, otherwise whisper.cpp.
WHISPER_BACKEND  = os.environ.get("WHISPER_BACKEND", "auto").lower()
_proj_whispercpp = Path(__file__).parent / "whispercpp"
WHISPERCPP_BIN   = os.environ.get("WHISPERCPP_BIN", str(_proj_whispercpp / "whisper-cli.exe"))
WHISPERCPP_MODEL = os.environ.get("WHISPERCPP_MODEL", str(_proj_whispercpp / "ggml-base.bin"))
# 多语种模型：当 WHISPERCPP_MODEL 是英文专用(*.en)时，非英文识别（如中文）改用它。
# ggml-base.bin 是多语种版（英文专用是 *.en.bin）。
WHISPERCPP_MODEL_MULTI = os.environ.get("WHISPERCPP_MODEL_MULTI", str(_proj_whispercpp / "ggml-base.bin"))
WHISPERCPP_LANG  = os.environ.get("WHISPERCPP_LANG", "en")
# whisper.cpp defaults to 4 threads; the Snapdragon X Elite has 12 cores, so on
# ARM64 (where the x64 binary runs emulated) more threads meaningfully speeds up
# transcription. Default 8 leaves headroom for the rest of the pipeline.
WHISPERCPP_THREADS = os.environ.get("WHISPERCPP_THREADS", "8")

# ── Thermal guard (GPU core-temp throttle) ──────────────────────────
# This 3080 has a history of falling off the bus ("失联", Xid 79) under
# sustained GPU load — a hardware/TIM issue, NOT pure core overheating (80°C is
# safe for the core; nvidia-smi does NOT expose the GDDR6X memory-junction temp,
# which is the real risk). We use core temp as the only available proxy: pause
# decoding between Whisper segments when it climbs, resume once it cools. This
# is a safety net, NOT a replacement for a host-side power limit / fan curve.
# Disable with WHISPER_THERMAL_GUARD=0. Only active on the cuda device.
WHISPER_THERMAL_GUARD     = os.environ.get("WHISPER_THERMAL_GUARD", "1") != "0"
WHISPER_TEMP_PAUSE_C      = int(os.environ.get("WHISPER_TEMP_PAUSE_C", "78"))   # pause at/above
WHISPER_TEMP_RESUME_C     = int(os.environ.get("WHISPER_TEMP_RESUME_C", "68"))  # resume below (hysteresis)
WHISPER_TEMP_POLL_SEC     = float(os.environ.get("WHISPER_TEMP_POLL_SEC", "4"))
WHISPER_TEMP_MAX_WAIT_SEC = int(os.environ.get("WHISPER_TEMP_MAX_WAIT_SEC", "120"))  # cap a single cooldown wait

_whisper_model = None
_whisper_lock = threading.Lock()          # guards _whisper_model / _busy / _last_used
_whisper_transcribe_lock = threading.Lock()  # serializes the actual model.transcribe()
_whisper_last_used = 0.0   # time.monotonic() of last transcription activity
_whisper_busy = 0          # transcriptions in flight; idle unloader skips when > 0
_whisper_unloader_started = False


def _setup_windows_cuda_dll_paths():
    """
    On Windows, faster-whisper needs cuBLAS + cuDNN DLLs in the loader's
    search path. The pip wheels `nvidia-cublas-cu12` / `nvidia-cudnn-cu12`
    drop the DLLs into `<site-packages>/nvidia/<pkg>/bin/`, but neither
    Python nor ctranslate2's native binary will find them by default.

    On Windows we need BOTH techniques:
    - `os.add_dll_directory()` for newer Python-managed DLL loads
    - prepending to `PATH` for ctranslate2's compiled binary, which uses
      the legacy `LoadLibrary` and only respects PATH

    No-op on non-Windows or when the nvidia wheels aren't installed.
    """
    if os.name != "nt":
        return
    if not WHISPER_ENABLED or WHISPER_DEVICE == "cpu":
        return

    nvidia_dirs = []
    for pkg_name in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            pkg = __import__(pkg_name, fromlist=[""])
        except ImportError:
            continue
        pkg_dir = os.path.dirname(pkg.__file__ or "")
        if not pkg_dir:
            continue
        # Windows: bin/, Linux: lib/
        for subdir in ("bin", "lib"):
            candidate = os.path.join(pkg_dir, subdir)
            if os.path.isdir(candidate):
                try:
                    os.add_dll_directory(candidate)
                except OSError:
                    pass
                nvidia_dirs.append(candidate)
                break

    if nvidia_dirs:
        # Critical: prepend to PATH so ctranslate2's LoadLibrary finds the DLLs
        os.environ["PATH"] = os.pathsep.join(nvidia_dirs) + os.pathsep + os.environ.get("PATH", "")


_setup_windows_cuda_dll_paths()

jobs      = {}
dub_tasks  = {}
burn_tasks = {}
slide_tasks = {}

# MTV job store
mtv_jobs: dict = {}
mtv_jobs_lock = threading.Lock()

# Max SSE iterations before force-closing (prevents leaked threads)
_SSE_MAX_ITERS = 600  # 10 minutes at 1s intervals

# Task retention: purge completed tasks older than this (seconds)
_TASK_TTL = 3600  # 1 hour


def escape_ass_path(path):
    """Escape ASS file path for ffmpeg -vf ass= on Windows (C: -> C\\:)."""
    p = str(path).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        p = p[0] + "\\:" + p[2:]
    return p


def job_cache_path(job_id):
    return UPLOAD_DIR / f"{job_id}_cache.json"

def save_job_cache(job_id):
    job = jobs.get(job_id)
    if not job:
        return
    cache = {k: v for k, v in job.items() if k not in ("_lock",)}
    try:
        with open(job_cache_path(job_id), "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_job_cache(job_id):
    p = job_cache_path(job_id)
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def get_or_load_job(job_id):
    if job_id in jobs:
        return jobs[job_id]
    cache = load_job_cache(job_id)
    if cache:
        jobs[job_id] = cache
        return cache
    return None


def get_genai():
    from google import genai
    key = GEMINI_KEYS[_gemini_key_idx % len(GEMINI_KEYS)] if GEMINI_KEYS else GOOGLE_API_KEY
    return genai.Client(api_key=key)

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False

def get_video_duration(path):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
            capture_output=True, text=True
        )
        info = json.loads(r.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return 0

def check_nvenc():
    """Pure capability probe: is the h264_nvenc encoder compiled into ffmpeg?
    Says nothing about whether we SHOULD use it — see nvenc_enabled()."""
    try:
        r = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        return "h264_nvenc" in r.stdout
    except Exception:
        return False


# Global nvenc policy switch. The RTX 3080 on this box repeatedly drops off the
# PCIe bus ("GPU is lost") under sustained nvenc encoding — a thermal/hardware
# fault (PTM7950 not yet replaced). Default OFF: every encode path falls back to
# libx264 (CPU), which is stable and fast enough here. Re-enable with
# ALLOW_NVENC=1 in .env once the card's thermals are fixed.
ALLOW_NVENC = os.environ.get("ALLOW_NVENC", "0") == "1"


def nvenc_enabled():
    """Should encode paths use nvenc? Only when policy allows it AND ffmpeg can.
    All consumer sites gate on this (not check_nvenc) so flipping ALLOW_NVENC
    protects every encode path at once."""
    return ALLOW_NVENC and check_nvenc()

def seconds_to_ass(s):
    # Work in integer centiseconds so 2-decimal rounding can't push the seconds
    # field to "60.00" (an invalid ASS timestamp, e.g. for s=59.999); carry any
    # overflow into minutes/hours instead.
    cs = int(round(max(0.0, s) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    sec = cs / 100.0
    return f"{h}:{m:02d}:{sec:05.2f}"

_WRAP_PUNCT = "\uff0c\u3002\uff01\uff1f\u3001\uff1b\uff1a,.!?;:"
# \u8fd9\u4e9b\u5b57\u7b26\u201c\u4e4b\u540e\u201d\u662f\u4f18\u5148\u65ad\u884c\u70b9\uff1a\u6807\u70b9 + \u6536\u5c3e\u62ec\u53f7/\u5f15\u53f7\u3002\u82f1\u6587\u8fd8\u4f1a\u5728\u7a7a\u683c\uff08\u8bcd\u8fb9\u754c\uff09\u65ad\u884c\u3002
_BREAK_AFTER = _WRAP_PUNCT + ")]}\uff09\u3011\u300d\u300f\u201d"


def _char_width(ch):
    """\u5b57\u7b26\u663e\u793a\u5bbd\u5ea6\uff1aCJK/\u5168\u89d2\u7b97 2 \u5217\uff0c\u62c9\u4e01\u5b57\u6bcd\u3001\u6570\u5b57\u3001\u534a\u89d2\u6807\u70b9\u7b49\u7b97 1 \u5217\u3002
    \u7528\u4e1c\u4e9a\u5bbd\u5ea6\u5c5e\u6027\u5224\u65ad\uff0c\u6bd4\u201c\u6309\u5b57\u7b26\u6570\u201d\u66f4\u8d34\u8fd1\u5b9e\u9645\u5360\u5c4f\u5bbd\u5ea6\u2014\u2014\u8fd9\u6837 36 \u4e2a\u6c49\u5b57\u548c\u7ea6 72 \u4e2a
    \u82f1\u6587\u5b57\u6bcd\u5360\u540c\u6837\u7684\u884c\u5bbd\u9884\u7b97\uff0c\u82f1\u6587\u77ed\u53e5\u624d\u4e0d\u4f1a\u88ab\u5f53\u6210\u6c49\u5b57\u4e00\u6837\u8fc7\u65e9\u6298\u884c\u3002"""
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _disp_width(s):
    return sum(_char_width(ch) for ch in s)


def _tokenize_for_wrap(text):
    """\u5207\u6210\u53ef\u72ec\u7acb\u6392\u7248\u7684 token\uff1a\u6bcf\u4e2a CJK \u5b57\u7b26\u5355\u72ec\u6210 token\uff08CJK \u53ef\u5728\u4efb\u610f\u4e24\u5b57\u95f4\u65ad\u884c\uff09\uff1b
    \u8fde\u7eed\u7684\u975e\u7a7a\u767d\u975e CJK \u5b57\u7b26\uff08\u4e00\u4e2a\u82f1\u6587\u5355\u8bcd/\u6570\u5b57\u4e32\uff09\u4f5c\u4e3a\u6574\u4f53\u4fdd\u7559\u3001\u4e0d\u4ece\u4e2d\u95f4\u65ad\u5f00\uff1b\u7a7a\u683c\u4f5c
    \u4e3a\u5355\u72ec\u5206\u9694 token\u3002"""
    tokens, buf = [], []
    for ch in text:
        if ch.isspace():
            if buf:
                tokens.append("".join(buf)); buf = []
            tokens.append(" ")
        elif _char_width(ch) == 2:
            if buf:
                tokens.append("".join(buf)); buf = []
            tokens.append(ch)
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def _wrap_lines(text, col_budget):
    """\u6309\u663e\u793a\u5bbd\u5ea6\u8d2a\u5fc3\u6298\u884c\uff0c\u8fd4\u56de\u5404\u884c\u5b57\u7b26\u4e32\uff08\u5df2\u53bb\u6389\u884c\u9996\u5c3e\u7a7a\u683c\uff09\u3002
    CJK \u53ef\u5728\u4efb\u610f\u4f4d\u7f6e\u65ad\u884c\uff1b\u82f1\u6587\u5355\u8bcd\u6574\u4f53\u4fdd\u7559\u3001\u53ea\u5728\u7a7a\u683c\u5904\u65ad\u884c\u2014\u2014\u6ee1\u8db3\u201c\u975e\u7279\u6b8a\u60c5\u51b5\u4e0b\u82f1\u6587\u4e0d
    \u8de8\u884c\u201d\u3002\u4ec5\u5f53\u5355\u4e2a token \u672c\u8eab\u5c31\u8d85\u8fc7\u6574\u884c\u9884\u7b97\uff08\u8d85\u957f\u82f1\u6587\u5355\u8bcd/URL \u7b49\u6781\u7aef\u60c5\u51b5\uff09\u65f6\u624d\u786c\u5207\u3002"""
    col_budget = max(2, int(col_budget))
    lines, cur, cur_w = [], [], 0
    for tok in _tokenize_for_wrap(text):
        w = _disp_width(tok)
        if cur and cur_w + w > col_budget:          # \u653e\u4e0d\u4e0b\u4e86\uff1a\u5148\u7ed3\u675f\u5f53\u524d\u884c
            line = "".join(cur).strip()
            if line:
                lines.append(line)
            cur, cur_w = [], 0
            if tok == " ":
                continue                            # \u65ad\u884c\u5904\u7684\u7a7a\u683c\u4e22\u5f03
        if not cur and tok == " ":
            continue                                # \u4e0d\u5728\u884c\u9996\u7559\u7a7a\u683c
        if w > col_budget and tok != " ":           # \u5355 token \u6bd4\u6574\u884c\u8fd8\u5bbd\uff1a\u786c\u5207
            part, pw = "", 0
            for ch in tok:
                cw = _char_width(ch)
                if part and pw + cw > col_budget:
                    lines.append(part); part, pw = ch, cw
                else:
                    part += ch; pw += cw
            cur, cur_w = ([part], pw) if part else ([], 0)
            continue
        cur.append(tok); cur_w += w
    line = "".join(cur).strip()
    if line:
        lines.append(line)
    return lines


def wrap_subtitle_text(text, max_chars=36):
    # max_chars \u8868\u793a\u201c\u6bcf\u884c\u6700\u591a\u591a\u5c11\u4e2a\u6c49\u5b57\u201d\uff0c\u6362\u7b97\u6210\u663e\u793a\u5217\u9884\u7b97\uff08\u6c49\u5b57=2 \u5217\uff09\u3002
    col_budget = max(1, int(max_chars)) * 2
    lines = _wrap_lines(text, col_budget)
    return "\\N".join(lines) if lines else (text or "")


def split_long_subtitle(text, start, end, max_chars, max_lines=2):
    """\u628a\u4e00\u6761\u8fc7\u957f\u5b57\u5e55\u62c6\u6210\u591a\u6761\u201c\u987a\u5e8f\u64ad\u653e\u201d\u7684\u5b57\u5e55\uff0c\u4fdd\u8bc1\u6bcf\u6761\u7ecf wrap_subtitle_text \u6298\u884c\u540e
    \u4e0d\u8d85\u8fc7 max_lines \u884c\uff08\u9ed8\u8ba4 2 \u884c\uff09\uff0c\u907f\u514d\u4e00\u5c4f\u5806 4-5 \u884c\u76d6\u4f4f\u753b\u9762\u3002

    Whisper \u7247\u6bb5\u6309\u53e5/\u65f6\u95f4\u5408\u5e76\uff08~10-25s\uff09\uff0c\u4e00\u53e5\u957f\u82f1\u6587\u8bd1\u6210\u4e2d\u6587\u53ef\u80fd\u6298\u5230 4-5 \u884c\u76d6\u4f4f\u753b\u9762\u3002
    \u8d85\u957f\u65f6\u6309\u663e\u793a\u5bbd\u5ea6\u5207\u6210\u51e0\u6bb5\u987a\u5e8f\u64ad\u653e\uff08\u800c\u975e\u540c\u5c4f\u5806\u53e0\uff09\uff0c\u65ad\u70b9\u4f18\u5148\u843d\u5728\u6807\u70b9\u8fb9\u754c\u3001\u5176\u6b21\u82f1\u6587
    \u7a7a\u683c\uff08\u8bcd\u8fb9\u754c\uff09\uff0c\u65f6\u95f4\u6309\u5404\u6bb5\u663e\u793a\u5bbd\u5ea6\u6bd4\u4f8b\u5206\u914d\u3002"""
    text = (text or "").strip()
    # Guard: \u975e\u6b63\u7684 max_chars/max_lines\uff08\u6765\u81ea\u672a\u6821\u9a8c\u7684\u5ba2\u6237\u7aef\u53c2\u6570\uff09\u4f1a\u8ba9\u4e0a\u9650<=0\u3001while \u6c38\u4e0d
    # \u63a8\u8fdb -> \u70e7\u5f55\u7ebf\u7a0b 100% CPU \u7a7a\u8f6c\u3002\u94b3\u5230 >=1\u3002
    max_chars = max(1, int(max_chars))
    max_lines = max(1, int(max_lines))
    col_budget = max_chars * 2                       # \u6bcf\u884c\u663e\u793a\u5217\u9884\u7b97\uff08max_chars \u4e2a\u6c49\u5b57\uff09
    if len(_wrap_lines(text, col_budget)) <= max_lines:
        return [{"start": start, "end": end, "text": text}]

    cap_cols = col_budget * max_lines                # \u5355\u6761\u5b57\u5e55\u603b\u663e\u793a\u5bbd\u5ea6\u4e0a\u9650
    pieces, rest = [], text
    while _disp_width(rest) > cap_cols:
        # \u53d6\u4e0d\u8d85\u8fc7 cap_cols \u5217\u7684\u6700\u957f\u524d\u7f00\u3002
        cut, w = 0, 0
        for i, ch in enumerate(rest):
            cw = _char_width(ch)
            if w + cw > cap_cols:
                break
            w += cw
            cut = i + 1
        # \u56de\u9000\u5230\u6700\u8fd1\u7684\u6807\u70b9\u4e4b\u540e\uff08\u4f18\u5148\uff09\u6216\u7a7a\u683c\u5904\uff0c\u907f\u514d\u4ece\u82f1\u6587\u5355\u8bcd\u4e2d\u95f4\u65ad\u5f00\u3002
        brk = 0
        for i in range(cut, max(0, cut - 16), -1):
            ch = rest[i - 1]
            if ch in _BREAK_AFTER:
                brk = i; break
            if ch == " " and not brk:
                brk = i
        if brk:
            cut = brk
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        pieces.append(rest)
    pieces = [p for p in pieces if p]

    total = sum(_disp_width(p) for p in pieces) or 1
    span = max(0.0, end - start)
    cues = []
    t = start
    for p in pieces:
        seg = span * (_disp_width(p) / total)
        cues.append({"start": t, "end": t + seg, "text": p})
        t += seg
    cues[-1]["end"] = end   # guard against float drift
    return cues

def build_ass(subtitles, video_w=1920, video_h=1080, font_size_frac=0.04,
              max_chars=36, outline=3, sub_color="&H0000FFFF",
              x_frac=0.5, y_frac=0.92, ass_align=2):
    font_size = max(16, int(video_h * font_size_frac))
    margin_v  = max(5, int(video_h * (1 - y_frac) * 0.3))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_w}
PlayResY: {video_h}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Sans CJK SC,{font_size},{sub_color},&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,{outline},1.5,{ass_align},10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for sub in subtitles:
        # \u53bb\u6389\u53e5\u672b\u591a\u4f59\u53e5\u53f7/\u9017\u53f7\uff08\u517c\u987e\u5c1a\u672a\u5728\u6570\u636e\u5c42\u89c4\u6574\u7684\u65e7\u4efb\u52a1\uff09\u3002
        _src = strip_edge_punct(sub.get("translation", ""))
        # Split over-long cues into sequential \u2264max_lines (2-line) cues so a
        # single screen never stacks 4-5 lines over the picture.
        for cue in split_long_subtitle(_src,
                                       sub["start"], sub["end"], max_chars):
            if not cue["text"]:
                continue
            text = wrap_subtitle_text(cue["text"], max_chars)
            text = re.sub(r"^([\uff0c\u3002\uff01\uff1f\u3001,\.!?])", r" \1", text)
            start = seconds_to_ass(cue["start"])
            end   = seconds_to_ass(cue["end"])
            events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
    return header + "\n".join(events)


def add_log(job_id, msg, log_type="info"):
    job = jobs.get(job_id)
    if not job:
        return
    t = time.strftime("%H:%M:%S")
    job.setdefault("logs", []).append({"time": t, "msg": msg, "type": log_type})
    job["logs"] = job["logs"][-50:]


def set_model_warning(job_id, used_model, primary_model):
    """Surface a persistent banner when translation fell back off the chosen
    model. Set as a top-level job field so the SSE stream pushes it to the
    frontend (which renders it as a sticky warning, not a scrolling log line)."""
    job = jobs.get(job_id)
    if not job:
        return
    job["model_warning"] = (
        f"⚠️ 首选模型 {primary_model} 被限流/不可用，"
        f"本次已自动改用备选模型 {used_model} 完成翻译。"
    )
    add_log(job_id, f"未使用首选模型，备选：{used_model}", "warning")

def parse_subtitles_json(text):
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "subtitles" in data:
            return data["subtitles"]
    except Exception:
        pass
    try:
        idx = text.rfind("}")
        if idx > 0:
            for wrapper in [text[:idx+1] + "]", '{"subtitles":' + text[:idx+1] + "]"]:
                try:
                    data = json.loads(wrapper)
                    if isinstance(data, list):
                        return data
                    if isinstance(data, dict) and "subtitles" in data:
                        return data["subtitles"]
                except Exception:
                    pass
    except Exception:
        pass
    items = []
    pattern = re.compile(
        r'"start"\s*:\s*([\d.]+).*?"end"\s*:\s*([\d.]+).*?"translation"\s*:\s*"([^"]*)"',
        re.DOTALL
    )
    for m in pattern.finditer(text):
        items.append({
            "start": float(m.group(1)),
            "end":   float(m.group(2)),
            "original": "",
            "translation": m.group(3)
        })
    return items


# ── Audio-based timestamp refinement ───────────────────────────────
# Cheaper Gemini models (flash-lite etc.) produce timestamps that are
# semantically right but slightly off the actual speech onset/offset.
# That mismatch shows up as A/V sync drift in both burned subs and TTS dubbing.
# Pro handles this naturally; for cheaper models we snap subtitle boundaries
# to real speech boundaries detected via ffmpeg silencedetect.

def detect_silence_regions(video_path, noise_db=-28, min_silence_sec=0.35):
    """
    Run `ffmpeg -af silencedetect` to find silent regions in the video's audio.

    Returns a sorted list of (silence_start, silence_end) tuples in seconds.
    A "silence_end" marks where speech resumes; a "silence_start" marks where
    speech pauses. Returns [] on failure.

    Args:
        noise_db: threshold in dB. Lower (e.g. -35) = stricter (less detected
                  silence); higher (e.g. -22) = looser. -28 works well for
                  most narration-style videos.
        min_silence_sec: minimum gap duration to count as a silence boundary.
                  0.35s reliably catches breath pauses without over-segmenting.
    """
    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(video_path),
            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_sec}",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stderr  # ffmpeg writes filter output to stderr
    except Exception:
        return []

    silences = []
    cur_start = None
    for line in output.splitlines():
        m = re.search(r"silence_start:\s*([\d.]+)", line)
        if m:
            cur_start = float(m.group(1))
            continue
        m = re.search(r"silence_end:\s*([\d.]+)", line)
        if m and cur_start is not None:
            silences.append((cur_start, float(m.group(1))))
            cur_start = None
    return silences


def snap_subtitles_to_speech(subtitles, silences, tolerance=1.5,
                              min_duration=0.5):
    """
    Snap each subtitle's start/end to the nearest real speech boundary
    detected by silencedetect, within `tolerance` seconds.

    - `start` snaps to the nearest silence_end (= speech ONSET)
    - `end`   snaps to the nearest silence_start (= speech OFFSET)
    - Preserves non-overlap with the previous subtitle
    - Skips a snap that would shrink a subtitle below `min_duration`
    - If no silences detected, returns subtitles unchanged

    Returns (adjusted_subtitles, stats_dict).
    """
    if not silences or not subtitles:
        return subtitles, {"snapped_starts": 0, "snapped_ends": 0, "max_shift": 0.0}

    speech_onsets = sorted(s[1] for s in silences)   # silence_end values
    speech_offsets = sorted(s[0] for s in silences)  # silence_start values

    def nearest_within(candidates, target, max_dist):
        """Binary-search nearest candidate to target within max_dist."""
        import bisect
        idx = bisect.bisect_left(candidates, target)
        best = None
        best_dist = max_dist
        for i in (idx - 1, idx):
            if 0 <= i < len(candidates):
                d = abs(candidates[i] - target)
                if d <= best_dist:
                    best_dist = d
                    best = candidates[i]
        return best

    adjusted = []
    snapped_starts = snapped_ends = 0
    max_shift = 0.0

    for sub in subtitles:
        orig_start, orig_end = sub["start"], sub["end"]
        new_start, new_end = orig_start, orig_end

        # Snap start to nearest speech onset
        snap_s = nearest_within(speech_onsets, orig_start, tolerance)
        if snap_s is not None and abs(snap_s - orig_start) > 0.05:
            new_start = snap_s
            snapped_starts += 1
            max_shift = max(max_shift, abs(snap_s - orig_start))

        # Snap end to nearest speech offset
        snap_e = nearest_within(speech_offsets, orig_end, tolerance)
        if snap_e is not None and abs(snap_e - orig_end) > 0.05:
            new_end = snap_e
            snapped_ends += 1
            max_shift = max(max_shift, abs(snap_e - orig_end))

        # Don't overlap previous
        if adjusted and new_start < adjusted[-1]["end"]:
            new_start = adjusted[-1]["end"]

        # Reject snap if it would crush the subtitle below min_duration
        if new_end - new_start < min_duration:
            new_start, new_end = orig_start, orig_end
            # Adjust for overlap with previous (using original times)
            if adjusted and new_start < adjusted[-1]["end"]:
                new_start = adjusted[-1]["end"]
                if new_end - new_start < min_duration:
                    new_end = new_start + min_duration

        sub_new = dict(sub)
        sub_new["start"] = round(new_start, 3)
        sub_new["end"]   = round(new_end, 3)
        adjusted.append(sub_new)

    return adjusted, {
        "snapped_starts": snapped_starts,
        "snapped_ends": snapped_ends,
        "max_shift": round(max_shift, 2),
    }


# ── Whisper-based transcription (preferred path) ────────────────────
# faster-whisper produces precise word-level timestamps via VAD. We use it
# for timestamps + English ASR; Gemini is then used only for translation.
# This eliminates the A/V sync drift seen with Gemini-only timestamp estimates.

def get_whisper_model():
    """
    Lazy-load and cache the Whisper model. Returns None if faster-whisper
    isn't installed or the model fails to load. Thread-safe.
    """
    global _whisper_model, _whisper_last_used, _whisper_unloader_started
    if _whisper_model is not None:
        _whisper_last_used = time.monotonic()
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is not None:
            _whisper_last_used = time.monotonic()
            return _whisper_model
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            return None
        try:
            _whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE,
            )
        except Exception as e:
            print(f"[whisper] load failed on {WHISPER_DEVICE}/{WHISPER_COMPUTE}: {e}")
            # CPU fallback if CUDA load fails (e.g., GPU OOM from MeloTTS)
            if WHISPER_DEVICE != "cpu":
                try:
                    print("[whisper] retrying on CPU with int8...")
                    _whisper_model = WhisperModel(
                        WHISPER_MODEL_SIZE, device="cpu", compute_type="int8"
                    )
                except Exception as e2:
                    print(f"[whisper] CPU fallback also failed: {e2}")
                    return None
            else:
                return None
        _whisper_last_used = time.monotonic()
        if WHISPER_IDLE_UNLOAD_SEC > 0 and not _whisper_unloader_started:
            _whisper_unloader_started = True
            threading.Thread(target=_whisper_idle_unloader, daemon=True).start()
        return _whisper_model


def _whisper_idle_unloader():
    """
    Watchdog: drop the Whisper model after WHISPER_IDLE_UNLOAD_SEC without
    transcription activity, releasing GPU VRAM to the video-generation
    workloads sharing this card. Reload on next request is seconds (weights
    stay in the local HF cache). Never unloads while a transcription is
    running (_whisper_busy guard).
    """
    global _whisper_model
    import gc
    while True:
        time.sleep(30)
        with _whisper_lock:
            if (_whisper_model is None or _whisper_busy > 0
                    or time.monotonic() - _whisper_last_used < WHISPER_IDLE_UNLOAD_SEC):
                continue
            _whisper_model = None
        # ctranslate2 frees its CUDA allocations when the model object is
        # collected; empty_cache() additionally returns any torch-held cache
        # (MeloTTS) if torch happens to be installed.
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"[whisper] idle > {WHISPER_IDLE_UNLOAD_SEC}s — model unloaded, VRAM released", flush=True)


def _gpu_core_temp():
    """Current GPU core temperature in °C via nvidia-smi, or None if unavailable.
    NOTE: this is the CORE sensor only; nvidia-smi does not expose the GDDR6X
    memory-junction temp on consumer cards, which usually runs much hotter and
    is the more likely trigger for an off-the-bus event. Treat as a proxy."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip().splitlines()[0].strip())
    except Exception:
        return None


def _gpu_thermal_wait(log=None, where=""):
    """If the GPU core is at/above WHISPER_TEMP_PAUSE_C, block (sleeping, so the
    GPU actually idles and cools) until it drops below WHISPER_TEMP_RESUME_C or
    WHISPER_TEMP_MAX_WAIT_SEC elapses. No-op on cpu, when the guard is off, or
    when nvidia-smi is unavailable. Returns fast when already cool."""
    if not WHISPER_THERMAL_GUARD or WHISPER_DEVICE == "cpu":
        return
    t = _gpu_core_temp()
    if t is None or t < WHISPER_TEMP_PAUSE_C:
        return
    log = log or (lambda m: None)
    log(f"⚠ GPU {t}°C ≥ {WHISPER_TEMP_PAUSE_C}°C{where}，暂停转录降温至 ≤{WHISPER_TEMP_RESUME_C}°C...")
    waited = 0.0
    while t is not None and t > WHISPER_TEMP_RESUME_C:
        if waited >= WHISPER_TEMP_MAX_WAIT_SEC:
            log(f"⚠ 降温等待超过 {WHISPER_TEMP_MAX_WAIT_SEC}s（当前 {t}°C），继续转录")
            return
        time.sleep(WHISPER_TEMP_POLL_SEC)
        waited += WHISPER_TEMP_POLL_SEC
        t = _gpu_core_temp()
    log(f"GPU 已降至 {t}°C，恢复转录")


def _resolve_whisper_backend():
    """
    Decide which ASR backend to use. 'auto' prefers faster-whisper (GPU box)
    and falls back to whisper.cpp when the package can't be imported (ARM64).
    """
    if WHISPER_BACKEND in ("whispercpp", "faster-whisper"):
        return WHISPER_BACKEND
    try:
        import faster_whisper  # noqa: F401
        return "faster-whisper"
    except ImportError:
        return "whispercpp"


def transcribe_with_whispercpp(video_path, log_fn=None, language=None):
    """
    Transcribe via a prebuilt whisper.cpp CLI (CPU; runs under x64 emulation on
    ARM64). ffmpeg extracts 16kHz mono PCM, whisper-cli emits JSON, and we parse
    it into the same [{"start","end","text"}] contract as transcribe_with_whisper.
    Raises RuntimeError if the binary/model is missing or transcription fails.
    """
    log = log_fn or (lambda m: None)

    lang = language or WHISPERCPP_LANG
    bin_path = Path(WHISPERCPP_BIN)
    model_path = Path(WHISPERCPP_MODEL)
    # 英文专用模型（*.en.bin）无法转写中文等其它语言（会输出 "(speaking in foreign
    # language)"）。非英文时切到多语种模型（ggml-base.bin）。
    if lang and lang != "en" and model_path.stem.endswith(".en"):
        multi = Path(WHISPERCPP_MODEL_MULTI)
        if multi.exists():
            model_path = multi
            log(f"非英文（{lang}）识别，切换多语种模型 {multi.name}")
        else:
            raise RuntimeError(
                f"中文识别需要多语种 whisper 模型，但未找到 {multi}；"
                f"当前 {model_path.name} 是英文专用模型，无法转写 {lang}。")
    if not bin_path.exists():
        raise RuntimeError(f"whisper.cpp 可执行文件不存在: {bin_path}")
    if not model_path.exists():
        raise RuntimeError(f"whisper.cpp 模型不存在: {model_path}")

    # whisper.cpp needs 16kHz mono 16-bit PCM WAV; extract it with ffmpeg
    # (the same ffmpeg the rest of the pipeline already depends on).
    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "audio.wav"
        log("提取 16kHz 单声道音频 (ffmpeg)...")
        extract = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)],
            capture_output=True, text=True,
        )
        if extract.returncode != 0 or not wav_path.exists():
            raise RuntimeError(f"ffmpeg 音频提取失败: {(extract.stderr or '')[-500:]}")

        out_prefix = Path(td) / "out"
        log(f"whisper.cpp 转录中 (model={model_path.name})...")
        # -oj writes <out_prefix>.json ; -of sets the output prefix (no extension)
        proc = subprocess.run(
            [str(bin_path), "-m", str(model_path), "-f", str(wav_path),
             "-l", lang, "-t", WHISPERCPP_THREADS,
             "-oj", "-of", str(out_prefix)],
            capture_output=True, text=True,
        )
        json_path = Path(str(out_prefix) + ".json")
        if proc.returncode != 0 or not json_path.exists():
            raise RuntimeError(
                f"whisper.cpp 转录失败: {((proc.stderr or '') + (proc.stdout or ''))[-500:]}"
            )
        data = json.loads(json_path.read_text(encoding="utf-8"))

    result = []
    for seg in data.get("transcription", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        off = seg.get("offsets", {})
        result.append({
            "start": float(off.get("from", 0)) / 1000.0,
            "end":   float(off.get("to", 0)) / 1000.0,
            "text":  text,
        })
    return result


def transcribe_with_whisper(video_path, log_fn=None, language=None):
    """
    Transcribe video audio with Whisper using VAD for clean boundaries.
    Returns list of {"start": float, "end": float, "text": str}.
    language: ASR language code (e.g. "zh"); None -> default "en".
    Raises RuntimeError if Whisper is unavailable.

    Note: faster-whisper accepts video files directly — it uses ffmpeg
    under the hood to extract audio at 16kHz mono.
    """
    log = log_fn or (lambda m: None)

    # ARM64 / Surface Pro: faster-whisper (CUDA) is unavailable — delegate to
    # the whisper.cpp CLI backend (also skips the GPU thermal guard below).
    if _resolve_whisper_backend() == "whispercpp":
        return transcribe_with_whispercpp(video_path, log_fn, language=language)

    # IMPORTANT: log BEFORE get_whisper_model(), because that call triggers
    # the (potentially slow) first-time model download. If we log after,
    # the frontend shows nothing for minutes and looks frozen.
    global _whisper_model, _whisper_busy, _whisper_last_used
    if _whisper_model is None:
        log(f"\u9996\u6b21\u8fd0\u884c\uff1a\u4e0b\u8f7d Whisper \u6a21\u578b ({WHISPER_MODEL_SIZE}) \u7ea6 1.6GB\uff0c\u9700\u51e0\u5206\u949f\uff08\u4e0b\u8f7d\u540e\u7f13\u5b58\u5230 ~/.cache/huggingface\uff0c\u4e4b\u540e\u79d2\u7ea7\u52a0\u8f7d\uff09...")
    else:
        log(f"\u52a0\u8f7d Whisper \u6a21\u578b ({WHISPER_MODEL_SIZE} on {WHISPER_DEVICE})...")

    # Mark busy BEFORE loading the model so the idle unloader can never drop
    # it between load and use, or mid-transcription (segments is a lazy
    # generator \u2014 the model is in use until it's fully consumed below).
    with _whisper_lock:
        _whisper_busy += 1
    # faster-whisper's WhisperModel wraps a single ctranslate2 instance whose
    # transcribe() + lazy segments generator are NOT thread-safe; two concurrent
    # jobs on the shared model corrupt decoder state. Serialize the whole
    # transcribe-and-consume here (the _whisper_busy counter above only feeds the
    # idle unloader). Acquire before the try so finally never releases unheld.
    _whisper_transcribe_lock.acquire()
    try:
        model = get_whisper_model()
        if model is None:
            raise RuntimeError("faster-whisper \u672a\u5b89\u88c5\u6216\u6a21\u578b\u52a0\u8f7d\u5931\u8d25")

        log("\u6a21\u578b\u5c31\u7eea\uff0c\u5f00\u59cb\u8f6c\u5f55\u97f3\u9891...")

        # Don't start a fresh transcription onto an already-hot card.
        _gpu_thermal_wait(log, "\uff08\u5f00\u59cb\u524d\uff09")

        # VAD filter dramatically reduces hallucinated transcriptions on silence
        # and gives much cleaner segment boundaries.
        # - condition_on_previous_text: TRUE produces more coherent segments (cuts
        #   align with sentence boundaries better). End-of-video hallucinations
        #   are caught by the info.duration clamp + the strict thresholds below.
        # - vad_min_silence_ms=500 avoids breaking sentences at small breath pauses
        # - no_speech_threshold=0.7 (vs default 0.6) drops more "almost silent" segs
        # - log_prob_threshold=-0.8 (vs default -1.0) drops low-confidence transcriptions
        #   (these are usually where Whisper hallucinates plausible-sounding nonsense)
        # - hallucination_silence_threshold=2.0 skips silent regions longer than
        #   2s when a hallucination is detected (faster-whisper >=1.0 feature)
        transcribe_kwargs = dict(
            language=(language or "en"),
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=WHISPER_VAD_MIN_SILENCE_MS),
            condition_on_previous_text=WHISPER_CONDITION_ON_PREV,
            no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
            log_prob_threshold=WHISPER_LOG_PROB_THRESHOLD,
            compression_ratio_threshold=2.4,  # default, catches repetitive hallucinations
        )
        # hallucination_silence_threshold was added in faster-whisper 1.0+.
        # Try to use it; older SDKs will reject the kwarg.
        try:
            segments, info = model.transcribe(
                str(video_path),
                hallucination_silence_threshold=2.0,
                **transcribe_kwargs,
            )
        except TypeError:
            segments, info = model.transcribe(str(video_path), **transcribe_kwargs)

        result = []
        duration_limit = float(info.duration) if info.duration else None
        # segments is a lazy generator: the GPU decodes the next segment only as
        # we iterate. Checking temp between segments (throttled to once per
        # WHISPER_TEMP_POLL_SEC of wall time) lets us pause-and-cool mid-video
        # without spawning nvidia-smi for every segment.
        last_temp_check = time.monotonic()
        for seg in segments:
            now = time.monotonic()
            if now - last_temp_check >= WHISPER_TEMP_POLL_SEC:
                _gpu_thermal_wait(log, "（转录中）")
                last_temp_check = time.monotonic()
            text = (seg.text or "").strip()
            if not text:
                continue
            # Whisper sometimes hallucinates segments past the actual video end,
            # especially when the video ends in silence or music. Drop them.
            if duration_limit is not None and seg.start >= duration_limit:
                break
            result.append({
                "start": float(seg.start),
                "end":   min(float(seg.end), duration_limit) if duration_limit else float(seg.end),
                "text":  text,
            })
        return result
    finally:
        _whisper_transcribe_lock.release()
        with _whisper_lock:
            _whisper_busy -= 1
            _whisper_last_used = time.monotonic()


def group_whisper_segments(segments, target_sec=10.0, max_sec=25.0):
    """
    Merge consecutive Whisper segments into ~target_sec chunks, preferring
    sentence boundaries. Avoids both fragmenting (Whisper sometimes emits
    1-2s pieces) and over-long chunks that would force the dub to rush.

    IMPORTANT — keep these chunks COARSE (whole sentences, not sub-sentence).
    These groups feed BOTH translation AND dubbing: dubbing_task synthesizes
    one TTS clip per group and inserts silence at every group boundary (gap +
    slot padding). Splitting a sentence across groups therefore puts an
    unbearable pause MID-SENTENCE in the narration. The on-screen "max 2
    lines" requirement is handled separately and downstream by
    split_long_subtitle() at render time (build_ass), which never touches the
    dub audio — so do NOT shrink target_sec/max_sec to fix subtitle wrapping.
    """
    if not segments:
        return []

    SENTENCE_ENDS = (".", "!", "?", "\u3002", "\uff01", "\uff1f", "\u2026")
    grouped = []
    current = None

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if current is None:
            current = {"start": seg["start"], "end": seg["end"], "text": text}
            continue

        merged_dur = seg["end"] - current["start"]
        cur_dur    = current["end"] - current["start"]
        prev_ends_sentence = current["text"].rstrip().endswith(SENTENCE_ENDS)

        # Flush conditions:
        # 1. Would exceed max_sec → must split
        # 2. Already past target_sec AND prev chunk ended at a sentence → split
        if merged_dur > max_sec:
            grouped.append(current)
            current = {"start": seg["start"], "end": seg["end"], "text": text}
        elif cur_dur >= target_sec and prev_ends_sentence:
            grouped.append(current)
            current = {"start": seg["start"], "end": seg["end"], "text": text}
        else:
            current["end"] = seg["end"]
            current["text"] = current["text"] + " " + text

    if current:
        grouped.append(current)
    return grouped


# Sentence-ending punctuation, used to find sentence boundaries when
# re-segmenting cues for dubbing. Deliberately EXCLUDES the ellipsis "…":
# it's a mid-sentence pause/trailing-off, not a terminator, so "……" must not
# trigger a split (doing so orphaned a lone "…" fragment and broke the
# sentence across cues).
_SENTENCE_END_CHARS = "。！？．!?"

def _split_into_sentence_pieces(text):
    """Split text into pieces, each ending right after a sentence-ending
    punctuation. A trailing run with no terminator is kept as its own piece
    (it's a sentence that continues into the next cue)."""
    pieces, buf = [], ""
    for ch in text:
        buf += ch
        if ch in _SENTENCE_END_CHARS:
            pieces.append(buf)
            buf = ""
    if buf.strip():
        pieces.append(buf)
    return [p for p in pieces if p.strip()]


def resegment_by_sentence(subs):
    """For DUBBING ONLY: re-segment cues so each unit is exactly ONE sentence.

    dubbing_task synthesizes one TTS clip per unit and inserts silence at every
    unit boundary (the inter-unit `gap` plus trailing `slot` padding). The
    coarse Whisper grouping force-splits dense narration mid-sentence (cues hit
    max_sec before a sentence ends), so a sentence spanning two cues gets an
    unbearable pause MID-SENTENCE. But naively merging whole cues over-merges:
    a cue ending mid-sentence may itself contain several complete sentences, so
    chaining by cue end fuses many sentences into one giant block.

    Instead: (1) explode each cue into sentence pieces at internal 。！？,
    distributing the cue's time span proportionally by piece length; (2) merge
    consecutive pieces only until one ends a sentence. Result: one sentence per
    unit, continuous TTS (no mid-sentence pause), no mega-blocks. Operates on a
    copy; the stored subtitles JSON (subtitle-only burn / editing) is untouched."""
    # 1) explode cues into sentence pieces with proportional timing
    frags = []
    for s in subs:
        # Drop the cross-page「…」continuation markers: dubbing regroups cues into
        # whole sentences, so a mid-sentence page break no longer exists here —
        # the「…」would otherwise show mid-caption and make the TTS pause/voice it.
        text = (s.get("translation") or "").replace("…", "")
        if not text.strip():
            continue
        st, en = s["start"], s["end"]
        span = max(0.0, en - st)
        pieces = _split_into_sentence_pieces(text)
        total = sum(len(p) for p in pieces) or 1
        t = st
        first = len(frags)
        for p in pieces:
            seg = span * (len(p) / total)
            frags.append({
                "start": t, "end": t + seg, "translation": p,
                "ends_sentence": p.rstrip()[-1:] in _SENTENCE_END_CHARS,
                "original": s.get("original", ""),
            })
            t += seg
        frags[-1]["end"] = en  # snap last piece of the cue to the cue's end

    # 2) merge consecutive pieces until one ends a sentence
    units, cur = [], None
    for f in frags:
        if cur is None:
            cur = {"start": f["start"], "end": f["end"],
                   "translation": f["translation"], "original": f["original"]}
        else:
            cur["end"] = f["end"]
            cur["translation"] += f["translation"]
        if f["ends_sentence"]:
            units.append(cur)
            cur = None
    if cur is not None:
        units.append(cur)
    return units


def prepare_dub_segments(subs):
    """Turn subtitles into the exact one-sentence dubbing units dubbing_task
    synthesizes: same trailing-fragment drop, same sentence re-segmentation.

    Factored out so burn_dub_task can reproduce the IDENTICAL unit sequence and
    overlay freshly edited caption text onto the audio-aligned timeline by index
    — without re-dubbing. Edits that keep sentence structure (the common case:
    fixing wording) yield the same unit count, so the mapping is exact."""
    subs = list(subs or [])
    # Drop a trailing transcription fragment (same rule as dubbing_task) so the
    # unit count matches whether or not that stray last cue was present.
    if len(subs) >= 2:
        _last, _prev = subs[-1], subs[-2]
        _lt = (_last.get("original") or "").strip()
        _pt = (_prev.get("original") or "").strip()
        _ldur = (_last.get("end", 0) or 0) - (_last.get("start", 0) or 0)
        if (_lt and _ldur < 3.0
                and _lt[-1] not in ".?!。？！…"
                and _lt.lower().rstrip(" ,") in _pt.lower()):
            subs = subs[:-1]
    return resegment_by_sentence(subs)


def remap_units_onto_timeline(edited_units, aligned):
    """Place edited caption units on the audio-aligned timeline by CUMULATIVE
    CHARACTER position, so edits show on the real speech timeline even when the
    sentence boundaries no longer line up 1:1 (e.g. the user changed a 。 to a ，
    and merged two sentences → one fewer unit).

    Index-matching breaks on any structural edit and would silently fall back to
    the stale dubbed text. Instead we treat the aligned timeline as a piecewise
    char→time map (unit k owns its slice of the total text over [start,end]) and
    look up each edited unit's start/end char offset in it. Robust because an edit
    barely shifts character positions, so each caption lands on the speech that
    voices roughly the same characters."""
    segs, acc = [], 0
    for u in aligned:
        L = len(u.get("translation", ""))
        s = float(u.get("start", 0) or 0)
        e = float(u.get("end", s) or s)
        segs.append((acc, acc + L, s, e))
        acc += L
    if not segs:
        return edited_units
    last_t = segs[-1][3]
    def char_to_time(pos):
        if pos <= 0:
            return segs[0][2]
        for c0, c1, t0, t1 in segs:
            if c0 <= pos < c1:
                return t0 + ((pos - c0) / max(1, c1 - c0)) * (t1 - t0)
        return last_t
    out, cpos = [], 0
    for u in edited_units:
        L = len(u.get("translation", ""))
        t0 = char_to_time(cpos)
        t1 = char_to_time(cpos + L)
        if t1 <= t0:
            t1 = t0 + 0.3   # guard: every caption needs a positive on-screen span
        out.append({**u, "start": t0, "end": t1})
        cpos += L
    return out


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Structured-output schema forcing an OBJECT-wrapped array. A bare top-level
# array confuses tool-capable models like gemma4 (it emits a stray tool token);
# wrapping in {"subtitles": [...]} is reliable. Each item is {id, translation}.
_OLLAMA_TRANS_SCHEMA = {
    "type": "object",
    "properties": {
        "subtitles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"},
                               "translation": {"type": "string"}},
                "required": ["id", "translation"],
            },
        }
    },
    "required": ["subtitles"],
}


def _ollama_translate(model, prompt, log_fn=None):
    """Translate via a local Ollama model (e.g. gemma4:12b-it-qat). Returns
    {id: translation}.

    - think=False: gemma4 is a thinking model; left on, it emits its reasoning
      as the JSON content instead of the translation.
    - format=schema: forces the full {"subtitles":[{id,translation}]} array,
      otherwise the model may return a single object and drop items.
    - trust_env=False: the .env sets ALL_PROXY=socks5; without this httpx would
      tunnel the localhost call through the proxy and fail.
    """
    import httpx
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "format": _OLLAMA_TRANS_SCHEMA,
        "options": {"temperature": 0, "num_ctx": 16384},
        "messages": [{
            "role": "user",
            "content": prompt + "\n\n输出格式：{\"subtitles\": [{\"id\": 编号, \"translation\": \"中文\"}]}",
        }],
    }
    if log_fn:
        log_fn(f"调用本地 Ollama 模型 {model} 翻译…")
    # Thermal shield: don't kick off inference while the 3080 is already hot,
    # and let the monitor abort this in-flight call if temp crosses the hard
    # ceiling mid-inference (closing the client raises -> Gemini fallback).
    import gpu_guard
    gpu_guard.wait_until_cool(log_fn=log_fn, reason="Ollama 翻译")
    with httpx.Client(trust_env=False, timeout=600.0) as cli:
        gpu_guard.register_abortable(cli.close)
        try:
            resp = cli.post(f"{OLLAMA_BASE_URL}/api/chat", json=body)
            resp.raise_for_status()
            content = ((resp.json() or {}).get("message", {}) or {}).get("content", "")
        finally:
            gpu_guard.unregister_abortable(cli.close)
    data = json.loads(content)
    items = data.get("subtitles", []) if isinstance(data, dict) else data
    # Coerce id to int: the consumer looks up trans_map.get(i+1) with an int key,
    # so a string id ("1") would miss and fall back to English.
    out = {}
    for it in (items or []):
        if isinstance(it, dict) and "id" in it:
            try:
                out[int(it["id"])] = it.get("translation", "")
            except (ValueError, TypeError):
                pass
    return out


def gemini_key_pool():
    """统一 Gemini key 池:与 call_gemini_generate 轮换用的 GEMINI_KEYS 同源
    (项目本地 gemini_keys.txt + ~/news-brief/gemini_keys.txt + GOOGLE_API_KEY,
    见 _load_gemini_keys)。tts_gemini 等手动轮换处复用此列表,避免两套 key 池。"""
    return list(GEMINI_KEYS)


def _ollama_batch_translate(model, prompt_header, lines, log_fn=None):
    """熬粥式分批本地翻译,返回 {id: 译文}。供「用户选 Ollama」与「Gemini 全挂兜底」复用。"""
    _CH = max(1, int(os.environ.get("OLLAMA_TRANS_CHUNK", "5")))
    _nb = (len(lines) + _CH - 1) // _CH
    tm = {}
    for _bi in range(0, len(lines), _CH):
        _batch = lines[_bi:_bi + _CH]
        if log_fn:
            log_fn(f"本地翻译·熬粥分批：第 {_bi // _CH + 1}/{_nb} 批"
                   f"（{_bi + 1}-{_bi + len(_batch)}/{len(lines)} 段）")
        tm.update(_ollama_translate(model, prompt_header + "\n".join(_batch) + "\n", log_fn=log_fn))
    return tm


def _gemini_translate_to_map(ai_model, prompt, log_fn=None, on_fallback=None):
    """轮 Gemini key 池做翻译,返回 {id: 译文};所有 key 都不可用则返回 None(交上层兜底 Ollama)。
    每个 key 只试 ai_model 本身(fallback_chain=[],不升级到付费 Pro),靠 key 池 + Ollama 兜底。"""
    from google.genai import types as gtypes
    cfg = gtypes.GenerateContentConfig(
        max_output_tokens=65536,
        response_mime_type="application/json",
        temperature=0.0,
    )
    # call_gemini_generate 内部已轮 Gemini key 池;fallback_chain=[] 只用 ai_model 本身,不升付费 Pro。
    # 免费备选链(均免费档,不含付费 Pro):3.5-flash 撞 503/限流时先轮其它免费模型 × key 池,
    # 全免费档都挂了才由上层退 Ollama。避免「一个瞬时 503 就掉本地慢翻译」。
    try:
        response = call_gemini_generate(
            None, primary_model=ai_model,
            fallback_chain=["gemini-2.5-flash", "gemini-3.1-flash-lite"],
            contents=[gtypes.Part.from_text(text=prompt)],
            config=cfg, log_fn=log_fn, on_fallback=on_fallback,
        )
    except Exception as e:
        if log_fn:
            log_fn(f"Gemini key 池全部失败：{str(e)[:90]}")
        return None
    raw = (response.text or "").strip()
    code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if code_match:
        raw = code_match.group(1).strip()
    try:
        translations = json.loads(raw)
    except json.JSONDecodeError:
        arr_match = re.search(r'(\[[\s\S]*\])', raw)
        if not arr_match:
            raise ValueError(f"Gemini 翻译返回无法解析：{raw[:200]}")
        translations = json.loads(arr_match.group(1))
    if isinstance(translations, dict):
        translations = (translations.get("subtitles") or translations.get("data")
                        or next((v for v in translations.values() if isinstance(v, list)), []))
    trans_map = {}
    for item in (translations or []):
        if isinstance(item, dict) and "id" in item:
            try:
                trans_map[int(item["id"])] = item.get("translation", "")
            except (ValueError, TypeError):
                pass
    return trans_map


def translate_segments_via_gemini(segments, client, ai_model, log_fn=None,
                                  on_fallback=None):
    """
    Batch-translate English segments to Chinese. Returns a list of subtitle
    dicts {"start", "end", "original", "translation"} using Whisper's original
    timestamps unchanged. Uses Gemini, or a local Ollama model when ai_model is
    prefixed "ollama:" (falling back to Gemini if Ollama is unreachable).
    """
    if not segments:
        return []

    from google.genai import types as gtypes

    # Numbered list with duration hint, so Gemini knows the char budget
    lines = []
    for i, seg in enumerate(segments):
        dur = seg["end"] - seg["start"]
        lines.append(f"[{i+1}] ({dur:.1f}s) {seg['text']}")
    numbered = "\n".join(lines)

    prompt = (
        "\u8bf7\u5c06\u4ee5\u4e0b\u82f1\u6587\u7247\u6bb5\u7ffb\u8bd1\u4e3a\u6d41\u7545\u81ea\u7136\u7684\u7b80\u4f53\u4e2d\u6587\u89e3\u8bf4\u3002\n\n"
        "\u4e25\u683c\u8981\u6c42\uff1a\n"
        "1. \u8f93\u51fa JSON \u6570\u7ec4\uff0c\u6bcf\u9879\u683c\u5f0f\uff1a{\"id\": \u7f16\u53f7, \"translation\": \"\u4e2d\u6587\"}\n"
        "2. id \u4e0e\u8f93\u5165\u7f16\u53f7\u4e00\u4e00\u5bf9\u5e94\uff0c\u4e0d\u8981\u9057\u6f0f\u4efb\u4f55\u4e00\u6761\n"
        "3. \u26a0\ufe0f \u8d85\u5173\u952e\uff1a\u6bcf\u6761\u4e2d\u6587\u7ffb\u8bd1\u5fc5\u987b\u4ec5\u4ec5\u5bf9\u5e94\u8be5\u6761\u82f1\u6587\u7684\u8bed\u4e49\u8303\u56f4\u3002\n"
        "   \u4e25\u7981\u5c06\u4e0a\u6bb5\u672a\u8bf4\u5b8c\u7684\u534a\u53e5\u632a\u5230\u4e0b\u6bb5\uff0c\u4e5f\u4e25\u7981\u5c06\u4e0b\u6bb5\u7684\u5185\u5bb9\u63d0\u524d\u5230\u4e0a\u6bb5\u3002\n"
        "   \u5373\u4f7f\u4e0a\u4e0b\u6587\u8bfb\u8d77\u6765\u4e0d\u987a\u3001\u4e2d\u6587\u4e0d\u7b26\u5408\u4e60\u60ef\uff0c\u4e5f\u5fc5\u987b\u4fdd\u6301\u6bcf\u6bb5\u8fb9\u754c\u4e25\u683c\u4e00\u81f4\u2014\u2014\u8fd9\u662f\u5b57\u5e55\u4e0e\u753b\u9762\u540c\u6b65\u7684\u786c\u6027\u8981\u6c42\u3002\n"
        "   \u5982\u679c\u67d0\u6bb5\u82f1\u6587\u672c\u8eab\u5c31\u662f\u4e0d\u5b8c\u6574\u7684\u534a\u53e5\uff0c\u4e2d\u6587\u4e5f\u8bd1\u6210\u4e0d\u5b8c\u6574\u7684\u534a\u53e5\u3002\n"
        "4. \u4e2d\u6587\u9605\u8bfb\u901f\u5ea6\u7ea6 4.5 \u5b57/\u79d2\uff0c\u6bcf\u6761\u4e2d\u6587\u5b57\u6570\u4e25\u683c\u63a7\u5236\u5728 (\u65f6\u957f\u79d2 \u00d7 4.5) \u4ee5\u5185\n"
        "   \u4f8b\uff1a5s \u6bb5 \u2264 22 \u5b57\uff0c10s \u6bb5 \u2264 45 \u5b57\uff0c20s \u6bb5 \u2264 90 \u5b57\n"
        "5. \u7ffb\u8bd1\u53ef\u610f\u8bd1\u3001\u53ef\u6982\u62ec\uff0c\u4f46\u4ec5\u9650\u5f53\u524d\u6bb5\u843d\u5185\uff0c\u4e0d\u8981\u8de8\u6bb5\u91cd\u7ec4\u5185\u5bb9\n"
        "6. \u26a0\ufe0f \u5982\u679c\u67d0\u6bb5\u82f1\u6587\u770b\u8d77\u6765\u662f\uff1a\u4e71\u7801\u3001\u91cd\u590d\u54bc\u5495\u3001\u80cc\u666f\u97f3\u4e50\u62df\u58f0\u8bcd\u3001\u6216\u4e0e\u4e0a\u4e0b\u6587\u5b8c\u5168\u4e0d\u8fde\u8d2f\u7684\u788e\u7247\uff08\u5e38\u89c1\u4e8e\u89c6\u9891\u672b\u5c3e\u7684\u80cc\u666f\u58f0\uff09\uff0c\n"
        "   \u8bf7\u8f93\u51fa\u7a7a\u5b57\u7b26\u4e32 \"\" \u4f5c\u4e3a translation\uff0c\u800c\u4e0d\u8981\u786c\u7ffb\u8bd1\u3002\u5b89\u9759\u6bd4\u8c0e\u8a00\u597d\u3002\n"
        "7. \u53ea\u8f93\u51fa JSON\uff0c\u4e0d\u8981\u89e3\u91ca\u3001\u4e0d\u8981 markdown\n\n"
        f"\u8f93\u5165\uff08\u5171 {len(segments)} \u6bb5\uff0c\u683c\u5f0f\uff1a[\u7f16\u53f7] (\u65f6\u957f) \u82f1\u6587\uff09\uff1a\n"
        f"{numbered}\n"
    )

    # Local Ollama path (e.g. "ollama:gemma4:12b-it-qat"). Falls back to Gemini
    # if Ollama is down, keeping the Whisper timestamps (no re-transcription).
    trans_map = None
    if isinstance(ai_model, str) and ai_model.startswith("ollama:"):
        try:
            # 熬粥式分批：坏 TIM 的 3080 一次把 29 段塞进单次 12B 推理会瞬间冲到 82°C
            # （在 gpu_guard 65~88 死区里无人拦）。切成小批，每批复用指令头 + 全局编号，
            # 每批进 _ollama_translate 前都过一次 wait_until_cool 冷却闸门，单批推理短到
            # 冲不上 76、批间降回 55 再喂下一批——全程本地、不回退云端。批大小可调
            # OLLAMA_TRANS_CHUNK（默认 5；还热就调小）。
            _m = ai_model.split(":", 1)[1]
            trans_map = _ollama_batch_translate(_m, prompt[: -(len(numbered) + 1)], lines, log_fn=log_fn)
        except Exception as e:
            if log_fn:
                log_fn(f"\u672c\u5730 Ollama \u7ffb\u8bd1\u5931\u8d25\uff08{str(e)[:80]}\uff09\uff0c\u6539\u7528 Gemini")
            ai_model = "gemini-3.5-flash"   # real model for the Gemini fallback

    # An empty/partial Ollama result (valid JSON but missing/short 'subtitles',
    # or truncated output) would pass the `is None` gate below and leave every
    # segment as raw English. Treat <50% coverage as failure -> Gemini.
    if trans_map is not None and len(trans_map) < len(segments) * 0.5:
        if log_fn:
            log_fn(f"本地 Ollama 仅返回 {len(trans_map)}/{len(segments)} 条译文，改用 Gemini")
        trans_map = None
        ai_model = "gemini-3.5-flash"

    if trans_map is None:
        # 免费 Gemini key 池翻译;全部 key/模型不可用则返回 None,交下面兜底。
        trans_map = _gemini_translate_to_map(ai_model, prompt, log_fn=log_fn, on_fallback=on_fallback)
        if trans_map is None:
            # 终极兜底:退回本地 Ollama gemma4 熬粥(慢但免费、不报错、不掉云端)。
            if log_fn:
                log_fn("所有免费 Gemini key 都不可用 → 退回本地 Ollama gemma4 兜底翻译")
            if on_fallback:
                try:
                    on_fallback("ollama-gemma4(兜底)", ai_model)
                except Exception:
                    pass
            trans_map = _ollama_batch_translate("gemma4:12b-it-qat",
                                                prompt[: -(len(numbered) + 1)], lines, log_fn=log_fn)

    result = []
    for i, seg in enumerate(segments):
        result.append({
            "start":       round(seg["start"], 3),
            "end":         round(seg["end"], 3),
            "original":    seg["text"],
            "translation": trans_map.get(i + 1, seg["text"]),  # fallback: keep English
        })
    return result


def generate_subtitles_task(job_id, video_path, ai_model, transcribe_only=False,
                            transcribe_engine="gemini"):
    job = jobs[job_id]
    try:
        client = get_genai()
        duration = get_video_duration(video_path)
        job["duration"] = duration

        subtitles = None
        used_whisper = False
        transcribe_via = None   # 中文原文模式的实际引擎标签

        # ── Path 0: 中文原文模式 — 识别原文，不翻译 ──────────────────────
        # 引擎由前端选择：
        #   gemini  → Gemini 纯文本识别（文本最准，时间为块内近似，耗配额）
        #   whisper → 本地 whisper.cpp（lang=zh，VAD 精确时间轴，免配额，中文文本略逊）
        # 所选引擎为主、另一为备用（失败时自动回退）。识别文本同时写入 original 与
        # translation，下游 build_ass/预览/导出原样复用。
        if transcribe_only:
            job["status"]   = "transcribing"
            job["progress"] = 15
            job["message"]  = "识别原文字幕中（不翻译）..."

            def _via_whisper():
                if not WHISPER_ENABLED:
                    return None
                add_log(job_id, "本地 Whisper 识别（lang=zh，VAD 精确时间轴）...")
                segs = transcribe_with_whisper(
                    video_path, log_fn=lambda m: add_log(job_id, m), language="zh")
                if not segs:
                    return None
                subs = [{"start": s["start"], "end": s["end"],
                         "original": s["text"], "translation": s["text"]} for s in segs]
                add_log(job_id, f"Whisper 识别完成，{len(subs)} 段（VAD 精确时间）", "success")
                return subs, "whisper-zh"

            def _via_gemini():
                add_log(job_id, f"Gemini 纯文本识别（{TRANSCRIBE_MODEL}）...")
                subs = transcribe_to_subtitles(   # 内部已做繁→简 + 标点规整
                    video_path, GEMINI_KEYS, progress=lambda m: add_log(job_id, m))
                return (subs, "gemini-asr") if subs else None

            def _via_hybrid():
                # 混合：Whisper 出精确时间轴 + Gemini 听音频校正文本
                if not WHISPER_ENABLED:
                    return None
                add_log(job_id, "混合模式：本地 Whisper 出精确时间轴（lang=zh）...")
                segs = transcribe_with_whisper(
                    video_path, log_fn=lambda m: add_log(job_id, m), language="zh")
                if not segs:
                    return None
                add_log(job_id, f"Whisper {len(segs)} 段，调 Gemini 听音频校正文本（保留精确时间）...")
                subs = transcribe_hybrid(
                    video_path, segs, GEMINI_KEYS,
                    progress=lambda m: add_log(job_id, m))
                return (subs, "hybrid") if subs else None

            if transcribe_engine == "whisper":
                order = [_via_whisper, _via_gemini]
            elif transcribe_engine == "hybrid":
                order = [_via_hybrid, _via_gemini, _via_whisper]
            else:
                order = [_via_gemini, _via_whisper]
            raw_subs = None
            for idx, fn in enumerate(order):
                try:
                    res = fn()
                except Exception as e:
                    add_log(job_id, f"识别引擎失败：{e}", "warning")
                    res = None
                if res and res[0]:
                    raw_subs, transcribe_via = res
                    break
                if idx == 0:
                    add_log(job_id, "主引擎未产出，尝试备用引擎...", "warning")
            if not raw_subs:
                raise RuntimeError("识别失败：所选引擎与备用引擎都未产出字幕")

            # 统一后处理：繁→简 + 半角标点→全角 + 每屏≤2行 + 去首尾多余标点。
            split_subs = []
            for s in raw_subs:
                text = normalize_punct(to_simplified(s["translation"]))
                for p in split_long_subtitle(text, s["start"], s["end"],
                                             TRANSCRIBE_LINE_CHARS, max_lines=2):
                    txt = strip_edge_punct(p["text"])
                    if txt:
                        split_subs.append({"start": p["start"], "end": p["end"],
                                           "original": txt, "translation": txt})
            subtitles = split_subs
            add_log(job_id, f"识别完成，共 {len(subtitles)} 条原文字幕（每屏≤2行）", "success")

        # ── Path 1: Whisper local transcription + Gemini translation ────
        # Preferred when faster-whisper is installed. Whisper gives precise
        # timestamps via VAD; Gemini only translates text. No video upload
        # needed, no timestamp drift, no A/V sync issues.
        if WHISPER_ENABLED and subtitles is None:
            try:
                job["status"]   = "transcribing"
                job["progress"] = 15
                job["message"]  = "Whisper \u672c\u5730\u8f6c\u5f55..."
                add_log(job_id, f"\u4f7f\u7528 Whisper ({WHISPER_MODEL_SIZE} on {WHISPER_DEVICE}) \u672c\u5730\u8f6c\u5f55...")

                whisper_segs = transcribe_with_whisper(
                    video_path,
                    log_fn=lambda m: add_log(job_id, m),
                )
                if not whisper_segs:
                    raise RuntimeError("Whisper \u672a\u68c0\u6d4b\u5230\u8bed\u97f3\u5185\u5bb9")

                add_log(job_id, f"\u8f6c\u5f55\u5b8c\u6210\uff0c\u539f\u59cb {len(whisper_segs)} \u6bb5\uff0c\u6309\u53e5\u5b50\u8fb9\u754c\u5408\u5e76\u4e2d...")
                grouped = group_whisper_segments(whisper_segs)

                job["status"]   = "generating"
                job["progress"] = 60
                job["message"]  = f"\u7ffb\u8bd1 {len(grouped)} \u6bb5\u4e2d..."
                add_log(job_id, f"\u8c03\u7528 Gemini \u7ffb\u8bd1 {len(grouped)} \u6bb5\u6587\u672c\uff08\u65e0\u9700\u4e0a\u4f20\u89c6\u9891\uff09...")

                subtitles = translate_segments_via_gemini(
                    grouped, client, ai_model,
                    log_fn=lambda m: add_log(job_id, m, "warning"),
                    on_fallback=lambda used, primary: set_model_warning(job_id, used, primary),
                )
                used_whisper = True
                add_log(job_id, f"\u7ffb\u8bd1\u5b8c\u6210\uff0c\u65f6\u95f4\u6233\u6765\u81ea Whisper VAD\uff08\u9ad8\u7cbe\u5ea6\uff09", "success")

            except Exception as whisper_err:
                add_log(
                    job_id,
                    f"Whisper \u8def\u5f84\u5931\u8d25\uff0c\u56de\u9000\u5230 Gemini \u89c6\u9891\u6a21\u5f0f\uff1a{whisper_err}",
                    "warning",
                )
                subtitles = None

        # ── Path 2: Gemini-video fallback (original flow) ─────────────────
        if subtitles is None:
            job["status"]   = "uploading_gemini"
            job["progress"] = 10
            job["message"]  = "\u6b63\u5728\u4e0a\u4f20\u89c6\u9891\u5230 Gemini..."
            add_log(job_id, "\u4e0a\u4f20\u89c6\u9891\u5230 Gemini Files API...")

            from google.genai import types as gtypes
            with open(video_path, "rb") as f:
                file_ref = client.files.upload(
                    file=f,
                    config=gtypes.UploadFileConfig(mime_type="video/mp4")
                )

            job["status"]   = "gemini_processing"
            job["progress"] = 30
            job["message"]  = "Gemini \u5904\u7406\u4e2d..."
            add_log(job_id, "Gemini \u5206\u6790\u89c6\u9891\u4e2d...")

            file_ready = False
            for _ in range(60):
                fstate = client.files.get(name=file_ref.name)
                if fstate.state.name == "ACTIVE":
                    file_ready = True
                    break
                if fstate.state.name == "FAILED":
                    raise ValueError("Gemini \u6587\u4ef6\u5904\u7406\u5931\u8d25\uff0c\u8bf7\u91cd\u8bd5")
                time.sleep(3)
            if not file_ready:
                raise ValueError("Gemini \u6587\u4ef6\u5904\u7406\u8d85\u65f6\uff083\u5206\u949f\uff09\uff0c\u8bf7\u91cd\u8bd5")

            job["status"]   = "generating"
            job["progress"] = 50
            job["message"]  = "AI \u751f\u6210\u5b57\u5e55\u4e2d..."
            add_log(job_id, "AI \u751f\u6210\u4e2d\u6587\u5b57\u5e55...")

            target_count = int(duration / 10)
            seg_dur = duration / max(1, target_count)
            prompt = f"""Watch this video carefully and generate Chinese subtitle commentary with accurate timestamps.

Video duration: {duration:.1f} seconds
Target: ~{target_count} subtitle entries (video is {duration:.0f}s, roughly one entry per 10s)

CRITICAL RULES:
1. NEVER cut in the middle of a sentence \u2014 each subtitle must be a complete semantic unit.
   A single subtitle can be up to 30 seconds if needed to preserve meaning.
2. Timestamps must be real video timestamps in seconds (float). Last entry end \u2248 {duration:.1f}
3. Chinese narration speed is ~4.5 chars/sec. Strictly limit each translation to
   (time_slot_seconds \u00d7 4.5) Chinese characters so it can be read at normal speed.
   Examples: 5s slot \u2192 max 22 chars, 10s slot \u2192 max 45 chars, 20s slot \u2192 max 90 chars
4. Translate/summarize English speech into natural, fluent Chinese (simplified).

Return ONLY a valid JSON array, no markdown, no explanation:
[
  {{"start": 0.0, "end": 12.5, "slot_seconds": 12.5, "original": "...", "translation": "\u5b8c\u6574\u8bed\u4e49\u5355\u5143\uff0c\u4e0d\u8d85\u8fc756\u5b57"}},
  ...
]"""

            # Ollama has no video-upload path; this Gemini-only mode (used when
            # Whisper is unavailable) swaps a local model back to Gemini.
            if isinstance(ai_model, str) and ai_model.startswith("ollama:"):
                add_log(job_id, "视频模式不支持本地模型，改用 Gemini 3.5 Flash", "warning")
                ai_model = "gemini-3.5-flash"

            response = call_gemini_generate(
                client,
                primary_model=ai_model,
                contents=[
                    gtypes.Part.from_uri(file_uri=file_ref.uri, mime_type="video/mp4"),
                    gtypes.Part.from_text(text=prompt),
                ],
                config=gtypes.GenerateContentConfig(
                    max_output_tokens=65536,
                    temperature=0.0,  # greedy: reproducible subtitle output
                ),
                log_fn=lambda m: add_log(job_id, m, "warning"),
                on_fallback=lambda used, primary: set_model_warning(job_id, used, primary),
            )

            raw_text = response.text or ""
            code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_text)
            if code_match:
                raw_text = code_match.group(1)

            subtitles = parse_subtitles_json(raw_text.strip())
            # Normalize: the merge loop below indexes sub['start']/['end']/
            # ['translation'] directly, so require start/end and default-fill the
            # text fields — otherwise a truncated/garbled model entry raises
            # KeyError and fails the whole job.
            subtitles = [
                {**s, "translation": s.get("translation", ""), "original": s.get("original", "")}
                for s in subtitles
                if isinstance(s, dict) and "start" in s and "end" in s
                and (s.get("end", 0) - s.get("start", 0)) > 0
            ]

            # Iterative merge: short subs (<6s) joined to neighbors
            for _pass in range(20):
                merged = []
                changed = False
                for sub in subtitles:
                    dur = sub["end"] - sub["start"]
                    if merged and dur < 6 and (sub["start"] - merged[-1]["end"]) < 0.3:
                        merged[-1]["end"] = sub["end"]
                        merged[-1]["translation"] += sub["translation"]
                        merged[-1]["original"]    += " " + sub.get("original", "")
                        changed = True
                    else:
                        merged.append(sub)
                subtitles = merged
                if not changed:
                    break

            # Silence-snap A/V correction (only useful for Gemini's imprecise
            # timestamps; Whisper path skips this since its timestamps are
            # already accurate at the VAD level).
            try:
                add_log(job_id, "\u68c0\u6d4b\u8bed\u97f3\u8fb9\u754c\uff0c\u4fee\u6b63\u65f6\u95f4\u6233...")
                silences = detect_silence_regions(video_path)
                if silences:
                    subtitles, snap_stats = snap_subtitles_to_speech(subtitles, silences)
                    add_log(
                        job_id,
                        f"\u97f3\u753b\u540c\u6b65\u4fee\u6b63\uff1a"
                        f"{snap_stats['snapped_starts']} \u4e2a\u8d77\u70b9 / "
                        f"{snap_stats['snapped_ends']} \u4e2a\u7ec8\u70b9\u88ab\u8c03\u6574\uff0c"
                        f"\u6700\u5927\u504f\u79fb {snap_stats['max_shift']}s",
                        "success",
                    )
                else:
                    add_log(job_id, "\u672a\u68c0\u6d4b\u5230\u660e\u663e\u8bed\u97f3\u95f4\u9699\uff0c\u8df3\u8fc7\u540c\u6b65\u4fee\u6b63", "info")
            except Exception as snap_err:
                add_log(job_id, f"\u540c\u6b65\u4fee\u6b63\u8df3\u8fc7\uff1a{snap_err}", "warning")

            # Clean up the uploaded Gemini file
            try:
                client.files.delete(name=file_ref.name)
            except Exception:
                pass

        # ── Save + finalize (common to both paths) ─────────────────────────
        # 去掉每条字幕译文首尾多余标点（句末句号、断句逗号）——预览/列表/SRT/烧录一致。
        # 仅处理 translation（显示/烧录用）；中文原文模式 Path 0 已处理过，这里幂等。
        for s in subtitles:
            if s.get("translation"):
                s["translation"] = strip_edge_punct(s["translation"])
        # 跨页没讲完的半句补「…」衔接：原始字幕条视图（预览/列表/SRT/纯字幕烧录）里
        # 一眼看出"续下页"。句末已是终结标点（。！？）或已有「…」的不动；最后一条不补。
        # 配音路径走 resegment（会剥掉「…」并按整句重组），所以这里加的标记不影响配音。
        _n = len(subtitles)
        for _i, s in enumerate(subtitles):
            t = (s.get("translation") or "").rstrip()
            if t and _i < _n - 1 and t[-1] not in "。！？．!?…":
                t = t + "…"
            s["translation"] = t
        sub_path = UPLOAD_DIR / f"{job_id}_subtitles.json"
        with open(sub_path, "w", encoding="utf-8") as f:
            json.dump(subtitles, f, ensure_ascii=False, indent=2)

        job["subtitles"]        = subtitles
        job["status"]           = "completed"
        job["progress"]         = 100
        job["message"]          = f"\u5b8c\u6210\uff01\u5171\u751f\u6210 {len(subtitles)} \u6761\u5b57\u5e55"
        job["subtitle_preview"] = subtitles[:5]
        job["transcribed_with"] = (transcribe_via if transcribe_only
                                   else "whisper" if used_whisper else "gemini")
        add_log(job_id, f"\u5b57\u5e55\u751f\u6210\u5b8c\u6210\uff0c\u5171 {len(subtitles)} \u6761", "success")

        if subtitles:
            first_texts = "".join(s.get("translation", "") for s in subtitles[:3])
            job["title"] = first_texts[:20] if first_texts else job_id[:8]

        save_job_cache(job_id)

    except Exception as e:
        job["status"]   = "error"
        job["progress"] = 0
        job["message"]  = f"\u5904\u7406\u5931\u8d25\uff1a{str(e)}"
        add_log(job_id, f"\u9519\u8bef\uff1a{str(e)}", "error")


def calculate_global_speed(subtitles, chars_per_sec, max_speed=1.3):
    ratios = []
    for sub in subtitles:
        duration = sub["end"] - sub["start"]
        if duration <= 0:
            continue
        text = sub.get("translation", "")
        needed_cps = len(text) / duration
        ratios.append(needed_cps / chars_per_sec)
    if not ratios:
        return 1.0
    return min(max(ratios) * 1.05, max_speed)


# TTS pacing config (env-tunable)
# target_fill=0.85 means TTS audio aims to fill 85% of each subtitle slot,
# leaving ~15% as natural trailing silence. Most segments will land at
# min_speed=1.0 (natural pace); only tight segments bump up toward max.
TTS_TARGET_FILL = float(os.environ.get("TTS_TARGET_FILL", "0.85"))
# Narrow, two-sided speed band keeps the perceived narration rate even.
# min<1.0 lets loose segments stretch slightly (fewer/shorter trailing
# silences) instead of speaking at 1.0x then sitting in dead air; max stays
# low so tight segments never audibly "rush". Old 1.0/1.3 band swung 30%
# segment-to-segment, which is what sounded like "时快时慢".
TTS_MIN_SPEED   = float(os.environ.get("TTS_MIN_SPEED",   "0.95"))
TTS_MAX_SPEED   = float(os.environ.get("TTS_MAX_SPEED",   "1.12"))
# Cap on trailing silence padded after an under-running segment. Prevents the
# 2-3s "cold gaps" (we measured up to 3.7s) that make pacing feel draggy.
# Surplus is moved to a single tail silence so the video isn't truncated.
TTS_MAX_GAP     = float(os.environ.get("TTS_MAX_GAP",     "0.8"))
# A Chinese dub is usually longer than the source video, so the whole track can
# overrun the last frame. When it does, speed the FULL track up uniformly to land
# exactly on the video's end (caption timeline scaled by the same factor). Uniform
# stretch is imperceptible and never sounds like "时快时慢"; capped here so we never
# rush the voice — any residue past the cap stays as a short freeze-frame tail.
TTS_FIT_MAX_SPEED = float(os.environ.get("TTS_FIT_MAX_SPEED", "1.25"))


def calculate_segment_speed(text, slot_seconds, base_chars_per_sec,
                            target_fill=None, min_speed=None, max_speed=None):
    """
    Per-segment TTS speed for natural, even pacing.

    Fixes the "rush-then-long-silence" pattern caused by using a single
    global speed driven by the worst-case segment. Each segment instead
    picks its own minimum speed needed to fit `target_fill` of its slot.

    Returns a speed clamped to the narrow [min_speed, max_speed] band:
      - loose segments are slowed toward `min_speed` (default 0.95x) so they
        stretch to fill more of their slot instead of finishing early and
        leaving a long trailing silence,
      - tight segments are sped toward `max_speed` (default 1.12x),
      - comfortable segments land in between at their natural fit ratio.
    Because the band is narrow, adjacent segments never differ enough to be
    heard as the narration speeding up and slowing down.
    """
    if target_fill is None: target_fill = TTS_TARGET_FILL
    if min_speed   is None: min_speed   = TTS_MIN_SPEED
    if max_speed   is None: max_speed   = TTS_MAX_SPEED

    if not text or slot_seconds <= 0 or base_chars_per_sec <= 0:
        return 1.0
    target_duration  = slot_seconds * target_fill
    natural_duration = len(text) / base_chars_per_sec
    raw = natural_duration / target_duration
    return min(max(raw, min_speed), max_speed)


def tts_edge(text, voice, output_path, rate="+0%"):
    import asyncio
    import edge_tts
    output_path = Path(output_path)
    tmp_mp3 = output_path.with_suffix(".tmp.mp3")
    async def _run():
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(str(tmp_mp3))
    asyncio.run(_run())
    subprocess.run([
        "ffmpeg", "-y", "-i", str(tmp_mp3),
        "-ar", "24000", "-ac", "1", "-sample_fmt", "s16",
        str(output_path)
    ], capture_output=True, check=True)
    tmp_mp3.unlink(missing_ok=True)

def tts_gemini(text, voice, output_path, client=None, model=None):
    from google.genai import types as gtypes
    from google import genai
    if model is None:
        model = "gemini-2.5-pro-preview-tts"
    cfg = gtypes.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=gtypes.SpeechConfig(
            voice_config=gtypes.VoiceConfig(
                prebuilt_voice_config=gtypes.PrebuiltVoiceConfig(voice_name=voice)
            )
        )
    )
    # Gemini key 池轮换:某 key 429(额度满)/失效 → 换下一把 key。
    keys = gemini_key_pool() or [None]
    response = None
    last_err = None
    for key in keys:
        cli = (client or get_genai()) if key is None else genai.Client(api_key=key)
        try:
            response = cli.models.generate_content(model=model, contents=text, config=cfg)
            break
        except Exception as e:
            last_err = e
            low = str(e).lower()
            if ("429" in str(e) or "resource_exhausted" in low or "api_key_invalid" in low
                    or "api key not valid" in low or "permission_denied" in low):
                continue
            raise
    if response is None:
        raise last_err or RuntimeError("Gemini TTS 所有 key 失败")
    audio_data = response.candidates[0].content.parts[0].inline_data.data
    import wave
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio_data)

def tts_fish(text, voice_id, output_path):
    import httpx
    resp = httpx.post(
        "https://api.fish.audio/v1/tts",
        headers={"Authorization": f"Bearer {FISH_API_KEY}"},
        json={"text": text, "reference_id": voice_id, "format": "wav"},
        timeout=60
    )
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(resp.content)

def tts_melotts(text, voice, output_path, speed=1.0):
    try:
        from melo.api import TTS
        model = TTS(language=voice, device="auto")
        speaker_ids = model.hps.data.spk2id
        speaker_id = list(speaker_ids.values())[0]
        model.tts_to_file(text, speaker_id, str(output_path), speed=speed)
    except ImportError:
        # MeloTTS (local CUDA/torch) isn't available on ARM64 / Surface Pro —
        # fall back to Edge TTS, preserving the requested speed as a rate adj.
        pct = int(round((speed - 1) * 100))
        tts_edge(text, "zh-CN-YunjianNeural", output_path, rate=f"{pct:+d}%")

def tts_openai(text, voice, output_path):
    import httpx
    api_key = OPENAI_API_KEY
    if not api_key:
        raise ValueError("\u672a\u8bbe\u7f6e OPENAI_API_KEY")
    resp = httpx.post(
        "https://api.openai.com/v1/audio/speech",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": "tts-1", "input": text, "voice": voice, "response_format": "wav"},
        timeout=60
    )
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(resp.content)

def detect_gemini_tts_speed(first_subtitle, voice, client):
    text = first_subtitle.get("translation", "")
    if not text:
        return 5.5
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        tts_gemini(text, voice, tmp.name, client)
        dur = get_wav_duration(tmp.name)
        if dur > 0:
            return len(text) / dur
    except Exception:
        pass
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return 5.5

def get_wav_duration(path):
    try:
        import wave
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0

def add_silence_wav(duration_sec, output_path, sample_rate=24000):
    import wave
    n_frames = int(duration_sec * sample_rate)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)

def concat_wav_files(wav_files, output_path):
    list_file = output_path.parent / f"_list_{uuid.uuid4().hex[:8]}.txt"
    try:
        with open(list_file, "w") as f:
            for w in wav_files:
                f.write(f"file '{w.resolve()}'\n")
        subprocess.run(
            # Re-encode rather than `-c copy`: TTS outputs differ in sample rate
            # (fish/openai return 44.1kHz, edge/gemini/silence are 24kHz). With
            # `-c copy` ffmpeg writes one header (the first input's) and stitches
            # the mismatched raw PCM, yielding wrong speed/pitch and broken
            # timeline alignment. Normalize everything to 24kHz mono s16.
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
             "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(output_path)],
            capture_output=True, check=True
        )
    finally:
        list_file.unlink(missing_ok=True)


# ── Voice routing ──────────────────────────────────────────────────
VOICE_ROUTE = {
    "edge_yuyang":   ("edge", "zh-CN-YunyangNeural"),
    "edge_xiaoxiao": ("edge", "zh-CN-XiaoxiaoNeural"),
    "edge_xiaoyi":   ("edge", "zh-CN-XiaoyiNeural"),
    "edge_yunxi":    ("edge", "zh-CN-YunxiNeural"),
    "edge_yunjian":  ("edge", "zh-CN-YunjianNeural"),
    "edge_tw":       ("edge", "zh-TW-HsiaoChenNeural"),
    "\u4e91\u626c": ("edge", "zh-CN-YunyangNeural"),
    "\u6653\u6653": ("edge", "zh-CN-XiaoxiaoNeural"),
    "\u6653\u4f0a": ("edge", "zh-CN-XiaoyiNeural"),
    "\u4e91\u5e0c": ("edge", "zh-CN-YunxiNeural"),
    "\u4e91\u5065": ("edge", "zh-CN-YunjianNeural"),
    "\u6653\u81fb": ("edge", "zh-CN-XiaozhenNeural"),
    "gemini_kore":   ("gemini", "Kore"),
    "gemini_charon": ("gemini", "Charon"),
    "gemini_fenrir": ("gemini", "Fenrir"),
    "gemini_puck":   ("gemini", "Puck"),
    "gemini_aoede":  ("gemini", "Aoede"),
    "gemini_leda":   ("gemini", "Leda"),
    "gemini_orus":   ("gemini", "Orus"),
    "gemini_zephyr": ("gemini", "Zephyr"),
    "Kore":   ("gemini", "Kore"),
    "Charon": ("gemini", "Charon"),
    "Fenrir": ("gemini", "Fenrir"),
    "Puck":   ("gemini", "Puck"),
    "Aoede":  ("gemini", "Aoede"),
    "Leda":   ("gemini", "Leda"),
    "Orus":   ("gemini", "Orus"),
    "Zephyr": ("gemini", "Zephyr"),
    "zh_female":  ("melo", "ZH"),
    "zh_male":    ("melo", "ZH"),
    "zh_en_mix":  ("melo", "ZH"),
    "en_female":  ("melo", "EN"),
    "en_male":    ("melo", "EN"),
    "openai_nova":    ("openai", "nova"),
    "openai_alloy":   ("openai", "alloy"),
    "openai_onyx":    ("openai", "onyx"),
    "openai_shimmer": ("openai", "shimmer"),
    "openai_echo":    ("openai", "echo"),
    "openai_fable":   ("openai", "fable"),
    "fish_1633917557": ("fish", "1633917557"),
}

MODEL_MAP = {
    "gemini":                "gemini-flash-lite-latest",
    "gemini-flash":          "gemini-flash-latest",
    "gemini-flash-lite":     "gemini-flash-lite-latest",
    "gemini-pro":            "gemini-pro-latest",
    "gemini-2.0-flash":      "gemini-flash-latest",
    "gemini-2.0-flash-lite": "gemini-flash-lite-latest",
    "gemini-2.5-flash":      "gemini-flash-latest",
    # gemini-3.1-flash-lite-preview was deprecated 5/11/26, shuts down 5/25/26.
    # GA version is now `gemini-3.1-flash-lite` (no -preview suffix).
    "gemini-3.1-flash":      "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
    "gemini-3.1-pro":        "gemini-3.1-pro-preview",
    # Gemini 3.5 Flash — GA'd at I/O 2026 (May 19). Stronger than 3.1 Pro on
    # agent/coding benchmarks. Dynamic thinking on by default; we configure
    # thinking_level via GEMINI_THINKING_LEVEL env var (default "high").
    # Free tier: 15 RPM / 1500 RPD. Now the default subtitle model.
    "gemini-3.5-flash":      "gemini-3.5-flash",
    # Local model via Ollama (offline, free, no rate limit). The "ollama:"
    # prefix routes translate_segments_via_gemini to the local HTTP path.
    "ollama-gemma4":         "ollama:gemma4:12b-it-qat",
}

# Fallback chain for subtitle generation. If the primary model fails (quota=0,
# not found, or persistent 429), fall through to the next.
#
# QUALITY FLOOR: this chain only contains models at least as strong as the
# default (gemini-3.5-flash). It deliberately NO LONGER drops to the weaker
# `flash-lite` family — that silent downgrade was the cause of "翻译降智":
# when 3.5-flash hit its free-tier 429, translation quietly continued on a
# much dumber lite model. Now a quota wall either lands on a pro-tier model
# (≥ quality) or, if those are also unavailable, raises a hard error the user
# sees — never a quietly worse translation. Any fallback that does happen is
# surfaced as a visible warning (see `on_fallback` / job["model_warning"]).
SUBTITLE_FALLBACK_CHAIN = [
    # 免费档备选(2026-06 查官方文档核实):付费 Pro 在免费 key 上配额=0、永远 429,当备选无意义;
    # gemini-2.0-flash 已被官方标记弃用(shutdown),不用。改用当前 GA 的免费 flash 轮换,
    # 全免费档都挂才退本地 Ollama。gemini-3.1-flash-lite 官方称 frontier-class,非旧版降智 lite。
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite",
]


def _apply_thinking_to_config(base_config, model):
    """
    Return a config with ThinkingConfig added if the model is Gemini 3.x.
    Falls back gracefully on older SDKs that don't support thinking_level.

    For translation tasks we use a low thinking level (configurable via
    GEMINI_THINKING_LEVEL env var) — gets the quality of the 3.5 family
    without the default MEDIUM-level latency tax (~10-15s extra TTFT).
    """
    if not model.startswith("gemini-3"):
        return base_config  # Pre-3 models don't accept thinking_config

    from google.genai import types as gtypes
    try:
        thinking_cfg = gtypes.ThinkingConfig(thinking_level=GEMINI_THINKING_LEVEL.lower())
    except (AttributeError, TypeError):
        # Older SDK without thinking_level support — skip silently
        return base_config

    # Rebuild the config with thinking_config merged in. The SDK's
    # GenerateContentConfig is a dataclass-like object; we copy its fields
    # and append thinking_config.
    if base_config is None:
        try:
            return gtypes.GenerateContentConfig(thinking_config=thinking_cfg)
        except (AttributeError, TypeError):
            return base_config

    try:
        # Most reliable: build a new config from the existing one's dict
        if hasattr(base_config, "model_dump"):
            kwargs = base_config.model_dump(exclude_none=True)
        elif hasattr(base_config, "dict"):
            kwargs = base_config.dict(exclude_none=True)
        else:
            kwargs = {k: v for k, v in vars(base_config).items() if v is not None}
        kwargs["thinking_config"] = thinking_cfg
        return gtypes.GenerateContentConfig(**kwargs)
    except Exception:
        # If anything weird happens, fall back to original config rather than fail
        return base_config


def call_gemini_generate(client, primary_model, contents, config=None,
                         fallback_chain=None, max_retries_per_model=2,
                         log_fn=None, on_fallback=None):
    """
    Wrapper for client.models.generate_content with auto-fallback.

    Behavior:
      - 429 with retryDelay → sleep + retry SAME model (up to max_retries_per_model)
      - 429 with `limit: 0` or 404/NOT_FOUND → skip to next model in chain
      - Any other exception → raise immediately

    `primary_model` is tried first, then each model in `fallback_chain` (skipping
    duplicates). If everything fails, raises the last exception.

    `on_fallback(used_model, primary_model)` is called once if the request
    ultimately succeeds on a model OTHER than `primary_model`, so callers can
    surface a prominent "didn't use your chosen model" warning rather than
    relying on the user spotting it scroll past in the log.
    """
    chain = [primary_model]
    for m in (fallback_chain or SUBTITLE_FALLBACK_CHAIN):
        if m not in chain:
            chain.append(m)

    log = log_fn or (lambda msg: print(f"[gemini-fallback] {msg}"))
    last_err = None

    global _gemini_key_idx
    # With a key pool (>1 key) a 429 or invalid-key rotates to the NEXT key
    # instantly (no sleep); expired/invalid keys are marked dead and skipped.
    # Single key → original sleep-and-retry behavior (unchanged).
    use_pool = len(GEMINI_KEYS) > 1
    # 单 key 路径用传入的 client;调用方(如 _gemini_translate_to_map)可能传 None
    # (本来指望走 key 池),此时按现有 key 兜底建一个,避免 None.models 崩溃。
    if not use_pool and client is None:
        from google import genai
        client = genai.Client(api_key=(GEMINI_KEYS[0] if GEMINI_KEYS else GOOGLE_API_KEY))

    def _quota_zero(msg):
        return ("NOT_FOUND" in msg or "not found" in msg.lower()
                or "limit: 0" in msg or "limit:0" in msg
                or "404" in msg[:120])

    def _key_dead(msg):
        return ("API_KEY_INVALID" in msg or "API_KEY_EXPIRED" in msg
                or "API key expired" in msg or "API key not valid" in msg)

    def _succeeded(model, resp):
        if model != primary_model:
            log(f"使用了备选模型 {model}（首选 {primary_model} 不可用）")
            if on_fallback:
                try:
                    on_fallback(model, primary_model)
                except Exception:
                    pass
        return resp

    for model in chain:
        # Apply thinking_level to Gemini 3.x models. Older models don't
        # accept this param and would error if we passed it. Done per-model
        # because the chain can include both 3.x and pre-3 models.
        eff_config = _apply_thinking_to_config(config, model)

        if use_pool:
            n = len(GEMINI_KEYS)
            # Try each key at most once for this model (skipping ones on cooldown);
            # the sequential index ensures every available key gets covered.
            for _ in range(n):
                with _gemini_key_lock:
                    idx = _gemini_key_idx % n
                    _gemini_key_idx += 1
                key = GEMINI_KEYS[idx]
                if _key_cooldown.get(key, 0) > time.time():
                    continue  # resting after a recent transient failure
                from google import genai
                try:
                    # Hold the client in a variable — an inline
                    # genai.Client(...).models.generate_content() temporary can be
                    # GC-closed mid-call ("client has been closed").
                    kclient = genai.Client(api_key=key)
                    resp = kclient.models.generate_content(
                        model=model, contents=contents, config=eff_config,
                    )
                    with _gemini_key_lock:
                        _gemini_key_idx = idx  # stick to the working key
                    _key_cooldown.pop(key, None)  # healthy → clear any cooldown
                    return _succeeded(model, resp)
                except Exception as e:
                    last_err = e
                    msg = str(e)
                    if _quota_zero(msg):
                        log(f"{model} 不可用（quota=0 或模型不存在），切换下一个模型")
                        break  # → next model
                    # invalid / 429 / 503 / overloaded → rest this key briefly and
                    # rotate. Keys are treated as valid-but-busy (not disabled).
                    if (_key_dead(msg) or "429" in msg or "RESOURCE_EXHAUSTED" in msg
                            or "503" in msg or "UNAVAILABLE" in msg
                            or "overloaded" in msg.lower()):
                        _key_cooldown[key] = time.time() + GEMINI_KEY_COOLDOWN
                        log(f"{model} key#{idx+1} 暂不可用，冷却 {GEMINI_KEY_COOLDOWN:.0f}s 后再用，换下一个 key")
                        continue  # → next key, no sleep
                    raise  # unknown error → bubble up
            continue  # all keys busy/cooling for this model → next model

        # Single-key path (original behavior).
        for attempt in range(max_retries_per_model):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=eff_config,
                )
                return _succeeded(model, resp)
            except Exception as e:
                last_err = e
                msg = str(e)

                # Permanently unavailable on this project → next model
                if _quota_zero(msg):
                    log(f"{model} 不可用（quota=0 或模型不存在），切换下一个")
                    break

                # Transient 429/503 → wait per retryDelay, then retry same model
                if ("429" in msg or "RESOURCE_EXHAUSTED" in msg
                        or "503" in msg or "UNAVAILABLE" in msg
                        or "overloaded" in msg.lower()):
                    delay = 15.0
                    m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg)
                    if m:
                        delay = min(float(m.group(1)) + 2, 60)
                    log(f"{model} 限流，{delay:.0f}s 后重试 (尝试 {attempt+1}/{max_retries_per_model})")
                    if attempt < max_retries_per_model - 1:
                        time.sleep(delay)
                        continue
                    log(f"{model} 重试用尽，切换下一个")
                    break

                # Unknown error → bubble up
                raise

    raise RuntimeError(f"所有 Gemini 模型都失败了，最后错误：{last_err}")

# TTS model display names (for reference)
TTS_MODEL_LABELS = {
    "gemini-2.5-pro-preview-tts":       "Gemini 2.5 Pro TTS",
    "gemini-2.5-flash-preview-tts":     "Gemini 2.5 Flash TTS",
    "gemini-3.1-flash-live-preview-tts": "Gemini 3.1 Flash Live TTS (低延迟)",
}


def dubbing_task(job_id, voice, tts_model="gemini-3.1-flash-live-preview-tts"):
    job = jobs.get(job_id)
    if not job:
        return

    dub_tasks[job_id] = {"status": "running", "progress": 0, "message": "\u521d\u59cb\u5316\u914d\u97f3..."}

    def update_dub(progress, message, status="running"):
        dub_tasks[job_id].update({"progress": progress, "message": message, "status": status})

    try:
        subtitles = job.get("subtitles", [])
        if not subtitles:
            sub_path = UPLOAD_DIR / f"{job_id}_subtitles.json"
            if sub_path.exists():
                with open(sub_path, encoding="utf-8") as f:
                    subtitles = json.load(f)
        if not subtitles:
            raise ValueError("\u6ca1\u6709\u627e\u5230\u5b57\u5e55\u6570\u636e")

        # Trailing-fragment drop + one-sentence re-segmentation live in
        # prepare_dub_segments, so burn_dub_task derives the SAME unit sequence
        # and can overlay edited captions onto the aligned audio timeline by
        # index. Display/editing keep the original subtitles JSON untouched.
        n_before = len(subtitles)
        subtitles = prepare_dub_segments(subtitles)
        if not subtitles:
            # resegment drops cues with blank translations, so a non-empty input
            # can become empty here -> guard before speed calc / TTS loop / concat
            # (gemini: subtitles[0] IndexError; others: empty-concat ffmpeg error).
            raise ValueError("没有可配音的文本（字幕译文为空）")
        update_dub(2, f"\u6309\u53e5\u91cd\u5207 {n_before}\u2192{len(subtitles)} \u53e5\uff08\u6bcf\u53e5\u8fde\u8bfb\uff0c\u907f\u514d\u53e5\u4e2d\u505c\u987f\uff09")

        audio_dir = UPLOAD_DIR / f"{job_id}_audio"
        # Always clear old audio to ensure fresh TTS with selected voice
        if audio_dir.exists():
            import shutil
            shutil.rmtree(audio_dir)
            update_dub(2, "清除旧音频，重新生成...")
        audio_dir.mkdir(exist_ok=True)
        voice_marker = audio_dir / "voice.txt"
        voice_marker.write_text(voice)

        # Resolve voice engine
        if voice in VOICE_ROUTE:
            engine, voice_name = VOICE_ROUTE[voice]
        elif ":" in voice:
            engine, voice_name = voice.split(":", 1)
        elif voice.startswith("zh-CN") or voice.startswith("zh-TW"):
            engine, voice_name = "edge", voice
        else:
            engine, voice_name = "edge", "zh-CN-YunjianNeural"

        update_dub(5, "\u8ba1\u7b97\u8bed\u901f...")
        client = None
        if engine == "gemini":
            client = get_genai()
            chars_per_sec = detect_gemini_tts_speed(subtitles[0], voice_name, client)
        elif engine == "melo":
            chars_per_sec = 4.5
        else:
            chars_per_sec = 5.5

        speed = calculate_global_speed(subtitles, chars_per_sec)

        # Compute per-segment speeds for engines that accept a speed parameter.
        # Each segment gets the minimum speed needed to fit ~target_fill of
        # its slot, so most segments speak at natural pace (1.0x) and only
        # tight ones speed up. Eliminates "rush-then-long-silence" pacing.
        supports_per_seg_speed = engine in ("edge", "melo")
        if supports_per_seg_speed:
            seg_speeds = [
                calculate_segment_speed(
                    s.get("translation", ""),
                    s["end"] - s["start"],
                    chars_per_sec,
                ) for s in subtitles
            ]
            avg_sp  = sum(seg_speeds) / len(seg_speeds) if seg_speeds else 1.0
            min_sp  = min(seg_speeds) if seg_speeds else 1.0
            max_sp  = max(seg_speeds) if seg_speeds else 1.0
            update_dub(10, f"\u8bed\u901f {min_sp:.2f}-{max_sp:.2f}x\uff08\u5e73\u5747 {avg_sp:.2f}x\uff09")
        else:
            seg_speeds = [speed] * len(subtitles)
            update_dub(10, f"\u5168\u5c40\u8bed\u901f: {speed:.2f}x")

        wav_segments = []
        prev_end = 0.0
        # `cursor` tracks the real position (seconds) inside merged.wav. A TTS
        # segment can run LONGER than its nominal slot (Gemini TTS has no speed
        # control, and edge/melo cap at max_speed), so the audio timeline drifts
        # later than the subtitles' nominal timestamps and the drift accumulates.
        # We record where each segment's speech actually plays and emit an aligned
        # caption timeline (dub_timeline) so the burned subtitle stays on screen
        # until the voice finishes, instead of vanishing at the nominal `end`.
        cursor = 0.0
        dub_timeline = []

        for i, sub in enumerate(subtitles):
            progress = 10 + int(i / len(subtitles) * 80)
            update_dub(progress, f"\u5408\u6210\u7b2c {i+1}/{len(subtitles)} \u6761\u5b57\u5e55...")

            text  = sub.get("translation", "")
            start = sub["start"]
            end   = sub["end"]

            # Re-anchor each segment to its ORIGINAL start time. Measuring the
            # gap from `cursor` (the real audio position) instead of `prev_end`
            # (the nominal one) makes this lead-in silence ABSORB any drift left
            # by earlier segments that overran their slot: it shrinks — or
            # vanishes — exactly enough to put this segment's speech back on the
            # video's timeline. Without it, a single overrun shifted every later
            # segment permanently later, so the dub fell progressively behind the
            # picture and the lag was worst at the end.
            gap = start - cursor
            if gap > 0.05:
                sil_path = audio_dir / f"sil_{i:04d}.wav"
                if not sil_path.exists():
                    add_silence_wav(gap, sil_path)
                wav_segments.append(sil_path)
                cursor += gap

            seg_path = audio_dir / f"seg_{i:04d}.wav"
            if not seg_path.exists() and text.strip():
                try:
                    seg_speed = seg_speeds[i]
                    if engine == "gemini":
                        tts_gemini(text, voice_name, seg_path, client, model=tts_model)
                        time.sleep(6)
                    elif engine == "edge":
                        # signed percent: e.g. -5% (slow), +0%, +12% (fast).
                        # edge_tts accepts negative rates, so loose segments can
                        # now be stretched instead of forced to +0%.
                        pct = int(round((seg_speed - 1) * 100))
                        rate_str = f"{pct:+d}%"
                        tts_edge(text, voice_name, seg_path, rate=rate_str)
                    elif engine == "fish":
                        tts_fish(text, voice_name, seg_path)
                    elif engine == "melo":
                        tts_melotts(text, voice_name, seg_path, speed=seg_speed)
                    elif engine == "openai":
                        tts_openai(text, voice_name, seg_path)
                except Exception as e:
                    update_dub(progress, f"TTS\u9519\u8bef(\u7b2c{i+1}\u6761): {str(e)[:60]}", "running")
                    add_silence_wav(end - start, seg_path)

            seg_start = cursor  # real audio offset where this caption's speech begins
            slot = end - start
            if seg_path.exists():
                dur  = get_wav_duration(seg_path)
                if dur < slot - 0.05:
                    # Speech shorter than slot: pad with silence, but CAP it at
                    # TTS_MAX_GAP so an over-long slot can't create a 2-3s cold
                    # gap. The trimmed surplus is recovered as one tail silence
                    # after the loop, so the video still isn't truncated.
                    gap_pad = min(slot - dur, TTS_MAX_GAP)
                    wav_segments.append(seg_path)
                    if gap_pad > 0.02:
                        sil_pad = audio_dir / f"pad_{i:04d}.wav"
                        add_silence_wav(gap_pad, sil_pad)
                        wav_segments.append(sil_pad)
                    cursor += dur + gap_pad
                else:
                    # Speech >= slot: it overruns; no padding. Cursor advances by
                    # the real duration, so everything after shifts later too.
                    wav_segments.append(seg_path)
                    cursor += dur
            else:
                empty = audio_dir / f"empty_{i:04d}.wav"
                add_silence_wav(slot, empty)
                wav_segments.append(empty)
                cursor += slot

            # Caption visible from where its speech starts until the block ends
            # (= speech end for overrun segments, slot end for padded ones), so it
            # never disappears before the voice does.
            dub_timeline.append({**sub, "start": seg_start, "end": cursor})

            prev_end = end

        # Capping per-segment silence above leaves the audio shorter than the
        # source video. Recover that slack as ONE trailing silence so the dub
        # spans the full video and `-shortest` won't chop the video tail.
        # (If speech overran instead, cursor >= video_dur and we add nothing;
        # the tpad freeze-frame branch below handles that case.)
        _vp = None
        for _ext in ("mp4", "mov", "avi", "mkv"):
            _p = UPLOAD_DIR / f"{job_id}.{_ext}"
            if _p.exists():
                _vp = _p
                break
        if _vp:
            _vdur = get_video_duration(_vp)
            if _vdur and cursor < _vdur - 0.1:
                tail_sil = audio_dir / "tail_sil.wav"
                add_silence_wav(_vdur - cursor, tail_sil)
                wav_segments.append(tail_sil)
                cursor = _vdur

        update_dub(92, "\u5408\u5e76\u97f3\u9891\u8f68\u9053...")
        merged_wav = audio_dir / "merged.wav"
        concat_wav_files(wav_segments, merged_wav)

        # Fit-to-video: if the assembled dub runs past the end of the source video
        # (common for zh dubs), compress the WHOLE track uniformly so the voice
        # finishes with the picture instead of spilling onto a long frozen frame.
        # One atempo over the entire track keeps every segment's relative timing,
        # so it stays in sync and doesn't reintroduce per-segment "\u65f6\u5feb\u65f6\u6162".
        # Capped at TTS_FIT_MAX_SPEED; if even the cap can't fit it, the leftover
        # overhang is handled by the freeze-frame (tpad) branch in the mux below.
        _vdur_fit = get_video_duration(_vp) if _vp else None
        merged_dur = get_wav_duration(merged_wav)
        if _vdur_fit and merged_dur and merged_dur > _vdur_fit + 0.15:
            fit_factor = min(merged_dur / _vdur_fit, TTS_FIT_MAX_SPEED)
            if fit_factor > 1.001:
                fitted_wav = audio_dir / "merged_fit.wav"
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", str(merged_wav),
                         "-filter:a", f"atempo={fit_factor:.5f}",
                         str(fitted_wav)],
                        capture_output=True, check=True,
                    )
                    merged_wav = fitted_wav
                    # Captions follow the audio, so scale their timeline by the
                    # same factor to keep each caption on its (now faster) speech.
                    for _seg in dub_timeline:
                        _seg["start"] /= fit_factor
                        _seg["end"]   /= fit_factor
                    update_dub(93, f"\u6574\u4f53\u63d0\u901f {fit_factor:.3f}x \u5bf9\u9f50\u7ed3\u5c3e")
                except subprocess.CalledProcessError:
                    pass  # atempo failed \u2192 keep original, freeze-frame tail covers it

        # Persist the audio-aligned caption timeline so the burn step shows each
        # subtitle exactly while its dubbed speech plays. Kept separate from the
        # nominal {job_id}_subtitles.json (which stays editable/translatable).
        try:
            dub_sub_path = UPLOAD_DIR / f"{job_id}_dubbed_subtitles.json"
            with open(dub_sub_path, "w", encoding="utf-8") as f:
                json.dump(dub_timeline, f, ensure_ascii=False)
            job["dub_subtitles"] = dub_timeline
        except Exception:
            pass

        update_dub(95, "\u5408\u5e76\u5230\u89c6\u9891...")
        video_path = None
        for ext in ("mp4", "mov", "avi", "mkv"):
            p = UPLOAD_DIR / f"{job_id}.{ext}"
            if p.exists():
                video_path = p
                break
        if not video_path:
            raise ValueError("video not found")

        dubbed_path = UPLOAD_DIR / f"{job_id}_dubbed.mp4"
        nvenc = nvenc_enabled()
        vcodec = "h264_nvenc" if nvenc else "libx264"

        # If accumulated TTS overruns made the audio longer than the source video,
        # `-shortest` alone would chop off the trailing speech. Freeze the last
        # video frame to cover the overhang so the final words are never cut.
        audio_dur = get_wav_duration(merged_wav)
        video_dur = get_video_duration(video_path)
        pad_tail  = audio_dur - video_dur if (audio_dur and video_dur) else 0.0

        def run_dub_ffmpeg(vc):
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(merged_wav),
            ]
            if pad_tail > 0.1:
                # tpad clones the last frame for pad_tail seconds; -shortest then
                # trims the now-longer video back down to the audio length.
                cmd += ["-vf", f"tpad=stop_mode=clone:stop_duration={pad_tail:.3f}"]
            cmd += [
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", vc, "-c:a", "aac", "-shortest",
                str(dubbed_path),
            ]
            subprocess.run(cmd, capture_output=True, check=True)

        try:
            run_dub_ffmpeg(vcodec)
        except subprocess.CalledProcessError:
            if vcodec != "libx264":
                run_dub_ffmpeg("libx264")

        job["dubbed_path"] = str(dubbed_path)
        update_dub(100, "\u914d\u97f3\u5b8c\u6210\uff01", "completed")
        dub_tasks[job_id]["_ts"] = time.time()

    except Exception as e:
        update_dub(0, f"\u914d\u97f3\u5931\u8d25\uff1a{str(e)}", "error")
        dub_tasks[job_id]["_ts"] = time.time()


def burn_dub_task(task_id, job_id, voice, x_frac, y_frac, font_size_frac,
                   ass_align, outline, sub_color, max_chars):
    burn_tasks[task_id] = {"status": "running", "message": "\u521d\u59cb\u5316..."}

    def update_burn(message, status="running"):
        burn_tasks[task_id].update({"status": status, "message": message})

    try:
        job = get_or_load_job(job_id)
        if not job:
            raise ValueError("\u4efb\u52a1\u4e0d\u5b58\u5728")

        dubbed_path = UPLOAD_DIR / f"{job_id}_dubbed.mp4"
        # If a dub already exists but was made with a different voice, it is stale
        # for this request — drop it (and its aligned-subtitle sidecar) so the
        # requested voice is honored instead of silently reusing the old audio.
        if dubbed_path.exists():
            _vm = UPLOAD_DIR / f"{job_id}_audio" / "voice.txt"
            _prev_voice = _vm.read_text().strip() if _vm.exists() else None
            if _prev_voice != voice:
                dubbed_path.unlink(missing_ok=True)
                (UPLOAD_DIR / f"{job_id}_dubbed_subtitles.json").unlink(missing_ok=True)
        if not dubbed_path.exists():
            update_burn("\u914d\u97f3\u4e2d...")
            dubbing_task(job_id, voice)
            if not dubbed_path.exists():
                raise ValueError("\u914d\u97f3\u5931\u8d25")

        update_burn("\u51c6\u5907\u5b57\u5e55\u6587\u4ef6...")
        subtitles = job.get("subtitles", [])
        if not subtitles:
            sub_path = UPLOAD_DIR / f"{job_id}_subtitles.json"
            if sub_path.exists():
                with open(sub_path, encoding="utf-8") as f:
                    subtitles = json.load(f)

        # Prefer the audio-aligned timeline produced by dubbing_task: its start/end
        # follow the real speech playback (including segments that overran their
        # slot), so captions stay on screen until the voice finishes. Falls back to
        # the nominal timestamps for dubs made before this file existed.
        #
        # But the aligned file carries the TEXT it was dubbed from, which is stale
        # after the user edits captions. So overlay the latest edits: re-derive the
        # same one-sentence units from the CURRENT subtitles and copy their text
        # onto the aligned timing, matched by index. Counts match whenever the edit
        # kept sentence structure (the common case: fixing wording) — then captions
        # show the new words on the right (audio-accurate) timeline. If the edit
        # changed structure so counts diverge, keep the aligned text rather than
        # drift onto nominal timestamps: staying in sync with the voice matters more
        # than reflecting a structural edit that would need a re-dub anyway.
        dub_sub_path = UPLOAD_DIR / f"{job_id}_dubbed_subtitles.json"
        if dub_sub_path.exists():
            try:
                with open(dub_sub_path, encoding="utf-8") as f:
                    aligned = json.load(f)
            except Exception:
                aligned = None
            if aligned:
                units = prepare_dub_segments(subtitles)
                if len(units) == len(aligned):
                    # Same sentence structure → exact 1:1 overlay of edited text
                    # onto each unit's real aligned start/end.
                    subtitles = [
                        {**a,
                         "translation": u.get("translation", a.get("translation", "")),
                         "original":    u.get("original", a.get("original", ""))}
                        for a, u in zip(aligned, units)
                    ]
                else:
                    # Structural edit (e.g. punctuation changed the sentence count):
                    # remap the edited units onto the aligned timeline by character
                    # position so the EDITED text still shows, in sync with the
                    # voice — never silently revert to the stale dubbed text.
                    subtitles = remap_units_onto_timeline(units, aligned)

        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "v:0", str(dubbed_path)],
            capture_output=True, text=True
        )
        info = json.loads(r.stdout)
        streams = info.get("streams", [{}])
        video_w = int(streams[0].get("width",  1920))
        video_h = int(streams[0].get("height", 1080))

        ass_content = build_ass(
            subtitles, video_w, video_h,
            font_size_frac=font_size_frac, max_chars=max_chars,
            outline=outline, sub_color=sub_color,
            x_frac=x_frac, y_frac=y_frac, ass_align=ass_align,
        )
        ass_path = UPLOAD_DIR / f"{task_id}.ass"
        ass_path.write_text(ass_content, encoding="utf-8")

        update_burn("\u70e7\u5f55\u5b57\u5e55...")
        output_path = UPLOAD_DIR / f"{task_id}_burned.mp4"
        nvenc = nvenc_enabled()
        vcodec = "h264_nvenc" if nvenc else "libx264"

        def run_burn(vc):
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(dubbed_path),
                "-vf", f"ass={escape_ass_path(ass_path)}",
                "-c:v", vc, "-c:a", "copy",
                str(output_path)
            ], capture_output=True, check=True)

        if output_path.exists():
            output_path.unlink()   # don't keep a stale burn if this run fails
        try:
            run_burn(vcodec)
        except subprocess.CalledProcessError:
            if vcodec == "libx264":
                raise          # no codec to fall back to — surface the failure
            run_burn("libx264")

        burn_tasks[task_id]["output"] = str(output_path)
        update_burn("\u5b8c\u6210\uff01", "completed")
        burn_tasks[task_id]["_ts"] = time.time()

    except Exception as e:
        update_burn(f"\u5931\u8d25\uff1a{str(e)}", "error")
        burn_tasks[task_id]["_ts"] = time.time()


def slide_dub_task(task_id, job_id, file_path, voice, tts_model=None,
                   burn_subs=True, sub_style=None):
    slide_tasks[task_id] = {"status": "running", "message": "\u521d\u59cb\u5316...",
                            "job_id": job_id, "_ts": time.time()}

    def update_slide(message, status="running"):
        slide_tasks[task_id].update({"status": status, "message": message})

    try:
        # Use job_id-based work_dir so re-runs resume from cached files
        work_dir = UPLOAD_DIR / f"{job_id}_slides"
        work_dir.mkdir(exist_ok=True)

        import glob as _glob
        import shutil as _shutil

        suffix = Path(file_path).suffix.lower()
        if suffix in (".pptx", ".ppt"):
            raise ValueError("PPT \u8bf7\u5148\u624b\u52a8\u8f6c\u6362\u4e3a PDF\uff0c\u518d\u4e0a\u4f20\u3002")
        elif suffix == ".pdf":
            pdf_path = work_dir / "input.pdf"
            # Detect if this is a new PDF by comparing MD5 hash
            new_hash = hashlib.md5(Path(file_path).read_bytes()).hexdigest()
            old_hash = hashlib.md5(pdf_path.read_bytes()).hexdigest() if pdf_path.exists() else ""
            is_new_pdf = (new_hash != old_hash)
            _shutil.copy2(file_path, pdf_path)  # always overwrite

            if is_new_pdf:
                # New PDF uploaded — clear ALL derived caches (PNGs, thumbs, narrations)
                update_slide("\u65b0 PDF\uff0c\u6e05\u9664\u65e7\u7f13\u5b58...")
                for old_png in work_dir.glob("page-*.png"):
                    old_png.unlink(missing_ok=True)
                thumbs_dir = work_dir / "thumbs"
                if thumbs_dir.exists():
                    _shutil.rmtree(thumbs_dir)
                narr_path = work_dir / "narrations.json"
                narr_path.unlink(missing_ok=True)
        else:
            raise ValueError(f"\u4e0d\u652f\u6301\u7684\u6587\u4ef6\u683c\u5f0f\uff1a{suffix}")

        # Always clear cached audio/clips to ensure fresh TTS with selected voice.
        for pattern in ["page_*_raw.wav", "silence_*.wav", "clip_*.mp4"]:
            for f in _glob.glob(str(work_dir / pattern)):
                Path(f).unlink(missing_ok=True)
        for old_f in ["merged.wav", "output.mp3", "output.mp4", "output_subbed.mp4",
                      "subs.ass", "clips_list.txt", "_voice.txt"]:
            (work_dir / old_f).unlink(missing_ok=True)

        update_slide("\u68c0\u67e5\u7f13\u5b58...")

        # PDF → PNG via pdftoppm (poppler)
        poppler_bin = (
            r"C:\Users\tanga\AppData\Local\Microsoft\WinGet\Packages"
            r"\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe"
            r"\poppler-25.07.0\Library\bin\pdftoppm.exe"
        )
        import shutil as _shutil2
        pdftoppm_cmd = poppler_bin if Path(poppler_bin).exists() else _shutil2.which("pdftoppm") or "pdftoppm"
        pdfinfo_cmd  = str(Path(poppler_bin).parent / "pdfinfo.exe") if Path(poppler_bin).exists() else "pdfinfo"

        # Get total page count for progress display
        total_pages = 0
        try:
            pi = subprocess.run([pdfinfo_cmd, str(pdf_path)], capture_output=True, text=True, timeout=10)
            for line in pi.stdout.splitlines():
                if line.lower().startswith("pages:"):
                    total_pages = int(line.split(":")[1].strip())
                    break
        except Exception:
            pass

        # Check if PNGs already exist (resume support) — only skip if count matches exactly
        existing_pngs = sorted(work_dir.glob("page-*.png"))
        if existing_pngs and total_pages > 0 and len(existing_pngs) == total_pages:
            update_slide(f"\u56fe\u7247\u5df2\u7f13\u5b58\uff08{len(existing_pngs)}\u9875\uff09\uff0c\u8df3\u8fc7\u8f6c\u6362")
        else:
            # Monitor PNG creation in background for progress
            stop_monitor = threading.Event()
            def _monitor():
                while not stop_monitor.is_set():
                    n = len(list(work_dir.glob("page-*.png")))
                    if total_pages:
                        update_slide(f"\u8f6c\u6362\u56fe\u7247 {n}/{total_pages} \u9875...")
                    else:
                        update_slide(f"\u8f6c\u6362\u56fe\u7247\uff08\u5df2\u5b8c\u6210 {n} \u9875\uff09...")
                    time.sleep(1)
            threading.Thread(target=_monitor, daemon=True).start()

            # try/finally so a pdftoppm failure (corrupt PDF, missing poppler,
            # disk full) still stops the monitor. Otherwise the daemon loops
            # forever calling update_slide(status="running"), permanently
            # clobbering the 'error' status the outer handler sets and leaking
            # the thread.
            try:
                subprocess.run(
                    [pdftoppm_cmd, "-png", "-r", "150", str(pdf_path), str(work_dir / "page")],
                    capture_output=True, check=True, timeout=600
                )
            finally:
                stop_monitor.set()

        page_images = sorted(work_dir.glob("page-*.png"))
        if not page_images:
            page_images = sorted(work_dir.glob("page-*.jpg")) + sorted(work_dir.glob("page-*.jpeg"))
        if not page_images:
            raise ValueError("PDF \u8f6c\u6362\u5931\u8d25\uff0c\u6ca1\u6709\u751f\u6210\u56fe\u7247")

        update_slide(f"\u5171 {len(page_images)} \u9875\uff0c\u6b63\u5728\u8bfb\u53d6\u5185\u5bb9...")

        topic_summary = ""
        if job_id != "standalone":
            job = get_or_load_job(job_id)
            if job and job.get("subtitles"):
                topic_summary = "".join(s.get("translation", "") for s in job["subtitles"])[:500]

        narrations_path = work_dir / "narrations.json"
        if narrations_path.exists():
            with open(narrations_path, encoding="utf-8") as f:
                existing = json.load(f)
            # If we already have all narrations, skip Gemini call
            narrations_list = [existing.get(str(i), "") for i in range(len(page_images))]
            if all(narrations_list):
                update_slide("\u8bb2\u89e3\u8bcd\u5df2\u7f13\u5b58\uff0c\u8df3\u8fc7 Gemini \u8c03\u7528")
            else:
                narrations_list = None
        else:
            narrations_list = None

        client = get_genai()
        from google.genai import types as gtypes

        if narrations_list is None:
            update_slide(f"Gemini \u5206\u6790\u5168\u90e8 {len(page_images)} \u9875\u5e7b\u706f\u7247...")
            # Send ALL pages in a single Gemini Vision call
            # Resize to JPEG 800px wide via ffmpeg (reduce 100MB → ~5MB total)
            thumb_dir = work_dir / "thumbs"
            thumb_dir.mkdir(exist_ok=True)

            contents = []
            for i, img_path in enumerate(page_images):
                thumb_path = thumb_dir / f"thumb_{i:04d}.jpg"
                if not thumb_path.exists():
                    try:
                        subprocess.run([
                            "ffmpeg", "-y", "-i", str(img_path),
                            "-vf", "scale=800:-2",
                            "-q:v", "4", str(thumb_path)
                        ], capture_output=True, check=True)
                    except Exception:
                        thumb_path = img_path  # fallback to original
                with open(thumb_path, "rb") as f:
                    img_data = f.read()
                mime = "image/jpeg" if str(thumb_path).endswith(".jpg") else "image/png"
                contents.append(gtypes.Part.from_bytes(data=img_data, mime_type=mime))

            no_bg = "\u65e0\u80cc\u666f\u4e3b\u9898"
            bg_line = topic_summary if topic_summary else no_bg
            prompt = (
                "\u4f60\u662f\u4e13\u4e1a\u6f14\u793a\u8bb2\u89e3\u5458\u3002\u4ee5\u4e0b\u662f\u8be5\u6f14\u793a\u7684\u80cc\u666f\u4e3b\u9898\uff1a\n"
                + bg_line + "\n\n"
                + "\u8bf7\u89c2\u5bdf\u6bcf\u5f20\u5e7b\u706f\u7247\u56fe\u7247\uff0c\u4e3a\u6bcf\u9875\u751f\u6210\u81ea\u7136\u6d41\u7545\u7684\u4e2d\u6587\u8bb2\u89e3\u8bcd\uff0850-80\u5b57\uff09\u3002\n"
                + "\u8981\u6c42\uff1a\u7ed3\u5408\u753b\u9762\u5185\u5bb9\uff0c\u7d27\u6263\u4e3b\u9898\uff0c\u8bb2\u89e3\u5458\u53e3\u543b\uff0c\u4e0d\u8981\u63cf\u8ff0\u56fe\u7247\u800c\u662f\u89e3\u8bf4\u5185\u5bb9\u3002\n"
                + f"\u8f93\u51fa JSON\uff1a[{{\"page\": 1, \"text\": \"...\"}}, {{\"page\": 2, \"text\": \"...\"}}, ...]\uff0c\u5171 {len(page_images)} \u9875\u3002\u4e0d\u8981\u5176\u4ed6\u5185\u5bb9\u3002"
            )
            contents.append(gtypes.Part.from_text(text=prompt))

            response = call_gemini_generate(
                client,
                primary_model="gemini-3.5-flash",
                contents=contents,
                config=gtypes.GenerateContentConfig(
                    max_output_tokens=65536,
                    response_mime_type="application/json",
                ),
                log_fn=lambda m: update_slide(m),
            )
            raw = response.text or ""
            code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
            if code_match:
                raw = code_match.group(1)
            raw = raw.strip()
            if not raw:
                raise ValueError(f"Gemini 返回空响应，无法生成旁白。请重试。(原始长度: {len(response.text or '')})")
            # Try to extract JSON array even if surrounded by extra text
            arr_match = re.search(r'(\[[\s\S]*\])', raw)
            if arr_match:
                raw = arr_match.group(1)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as je:
                print(f"[slide] JSON parse error: {je}\nRaw response ({len(raw)} chars):\n{raw[:500]}")
                raise ValueError(f"Gemini 返回了无效 JSON，请重试。错误：{je}")
            narrations_list = [""] * len(page_images)
            for item in parsed:
                idx = item.get("page", 1) - 1
                if 0 <= idx < len(page_images):
                    narrations_list[idx] = item.get("text", "")

            # Cache narrations
            cache = {str(i): t for i, t in enumerate(narrations_list)}
            with open(narrations_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

        audio_files = []
        if voice in VOICE_ROUTE:
            engine, voice_name = VOICE_ROUTE[voice]
        else:
            engine, voice_name = "edge", "zh-CN-YunjianNeural"

        for i, text in enumerate(narrations_list):
            update_slide(f"\u5408\u6210\u7b2c {i+1}/{len(page_images)} \u9875\u914d\u97f3...")
            audio_path = work_dir / f"page_{i:04d}_raw.wav"
            if not audio_path.exists() and text.strip():
                if engine == "gemini":
                    tts_gemini(text, voice_name, audio_path, client, model=tts_model)
                    time.sleep(6)
                elif engine == "edge":
                    tts_edge(text, voice_name, audio_path)
                elif engine == "fish":
                    tts_fish(text, voice_name, audio_path)
                elif engine == "melo":
                    tts_melotts(text, voice_name, audio_path)
                elif engine == "openai":
                    tts_openai(text, voice_name, audio_path)
            if audio_path.exists():
                audio_files.append(audio_path)
            sil_path = work_dir / f"silence_{i:04d}.wav"
            add_silence_wav(0.8, sil_path)
            audio_files.append(sil_path)

        update_slide("\u5408\u5e76\u97f3\u9891...")
        merged_wav = work_dir / "merged.wav"
        concat_wav_files(audio_files, merged_wav)

        output_mp3 = work_dir / "output.mp3"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(merged_wav),
            "-codec:a", "libmp3lame", "-q:a", "2", str(output_mp3)
        ], capture_output=True, check=True)
        slide_tasks[task_id]["output_mp3"] = str(output_mp3)

        update_slide("\u5408\u6210\u89c6\u9891\u5c42\u7247\u6bb5...")
        output_mp4 = work_dir / "output.mp4"
        clip_paths = []
        # Per-page subtitle cues on the FINAL concatenated timeline. Each page's
        # narration shows only while that page's speech plays (hidden during the
        # 0.8s trailing pad). `cum` tracks where each clip begins in output.mp4.
        slide_subs = []
        cum = 0.0

        for i, img_path in enumerate(page_images):
            update_slide(f"\u6e32\u67d3\u7b2c {i+1}/{len(page_images)} \u9875\u89c6\u9891...")
            audio_path = work_dir / f"page_{i:04d}_raw.wav"
            sil_path   = work_dir / f"silence_{i:04d}.wav"
            clip_path  = work_dir / f"clip_{i:04d}.mp4"

            sub_text   = narrations_list[i] if i < len(narrations_list) else ""
            speech_dur = get_video_duration(str(audio_path)) if audio_path.exists() else 0.0

            if clip_path.exists():
                clip_paths.append(clip_path)
                clip_dur = get_video_duration(str(clip_path)) or 0.0
                if sub_text.strip() and speech_dur > 0:
                    slide_subs.append({"translation": sub_text,
                                       "start": cum, "end": cum + speech_dur})
                cum += clip_dur
                continue

            # Merge audio + silence into one wav for this clip
            clip_wav = work_dir / f"clip_{i:04d}.wav"
            segs = []
            if audio_path.exists():
                segs.append(audio_path)
            if sil_path.exists():
                segs.append(sil_path)
            elif not segs:
                add_silence_wav(3.0, clip_wav)
                segs = [clip_wav]

            if len(segs) > 1:
                concat_wav_files(segs, clip_wav)
            elif segs and segs[0] != clip_wav:
                import shutil as _shutil
                _shutil.copy2(segs[0], clip_wav)

            # Per-page clip: still image + audio
            # Use explicit -t duration instead of -shortest for reliable sync
            clip_dur = get_video_duration(str(clip_wav))
            if clip_dur <= 0:
                clip_dur = 3.0
            # nvenc only when policy allows (see nvenc_enabled); otherwise go
            # straight to libx264 to avoid loading the fragile GPU. Either way
            # libx264 remains the fallback if the preferred encoder errors.
            _encoders = []
            if nvenc_enabled():
                _encoders.append(("h264_nvenc", ["-preset", "p1", "-tune", "ll"]))
            _encoders.append(("libx264", ["-preset", "ultrafast", "-tune", "stillimage"]))
            ffmpeg_ok = False
            for vcodec, extra in _encoders:
                r = subprocess.run([
                    "ffmpeg", "-y",
                    "-loop", "1", "-i", str(img_path),
                    "-i", str(clip_wav),
                    "-t", str(clip_dur),
                    "-vf", "scale='floor(iw/2)*2':'floor(ih/2)*2'",
                    "-c:v", vcodec, *extra,
                    "-c:a", "aac", "-pix_fmt", "yuv420p",
                    str(clip_path)
                ], capture_output=True)
                if r.returncode == 0:
                    ffmpeg_ok = True
                    break
            if not ffmpeg_ok:
                raise RuntimeError(f"ffmpeg clip {i} failed")
            clip_paths.append(clip_path)
            if sub_text.strip() and speech_dur > 0:
                slide_subs.append({"translation": sub_text, "start": cum,
                                   "end": cum + min(speech_dur, clip_dur)})
            cum += clip_dur

        # Concat all clips into final MP4
        update_slide("\u62fc\u63a5\u6240\u6709\u9875\u9762...")
        concat_list = work_dir / "clips_list.txt"
        with open(concat_list, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{cp.resolve()}'\n")
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c", "copy", "-reset_timestamps", "1", str(output_mp4)
        ], capture_output=True, check=True)

        # Burn per-page narration as subtitles onto the concatenated video.
        # Kept as a separate file so the clean output.mp4 stays available; the
        # download endpoint serves whatever output_mp4 points at. If the burn
        # fails (e.g. missing CJK font), fall back to the unsubbed video.
        final_mp4 = output_mp4
        if burn_subs and slide_subs:
            update_slide("\u70e7\u5f55\u5b57\u5e55...")
            vw, vh = 1920, 1080
            try:
                pr = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                     "-show_entries", "stream=width,height",
                     "-of", "csv=p=0:s=x", str(output_mp4)],
                    capture_output=True, text=True
                )
                wh = pr.stdout.strip().split("x")
                if len(wh) == 2:
                    vw, vh = int(wh[0]), int(wh[1])
            except Exception:
                pass
            st = sub_style or {}
            ass_path = work_dir / "subs.ass"
            ass_path.write_text(
                build_ass(
                    slide_subs, video_w=vw, video_h=vh,
                    font_size_frac=st.get("font_size_frac", 0.045),
                    sub_color=st.get("sub_color", "&H0000FFFF"),
                    y_frac=st.get("y_frac", 0.92),
                    ass_align=st.get("ass_align", 2),
                ),
                encoding="utf-8")
            subbed = work_dir / "output_subbed.mp4"
            if subbed.exists():
                subbed.unlink()

            # Burn with libx264 (CPU) ON PURPOSE, not nvenc: this is a single
            # sustained full-length re-encode, and on this box that load on the
            # thermally-marginal RTX 3080 has dropped the card off the PCIe bus
            # ("GPU is lost"). Slides are static images, so libx264 ultrafast is
            # quick and reliable. Per-page clip rendering still uses nvenc (short
            # bursts), but the long burn pass must not.
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(output_mp4),
                    "-vf", f"ass={escape_ass_path(str(ass_path))}",
                    "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-c:a", "copy", str(subbed)
                ], capture_output=True, check=True)
                # Guard against a truncated/partial output (e.g. ffmpeg killed
                # mid-write leaving a header-only file) being served as the result.
                if subbed.exists() and subbed.stat().st_size > 100_000:
                    final_mp4 = subbed
                else:
                    update_slide("\u5b57\u5e55\u70e7\u5f55\u8f93\u51fa\u5f02\u5e38\uff0c\u5df2\u8f93\u51fa\u65e0\u5b57\u5e55\u89c6\u9891")
            except Exception as e:
                update_slide(f"\u5b57\u5e55\u70e7\u5f55\u5931\u8d25\uff0c\u8f93\u51fa\u65e0\u5b57\u5e55\u89c6\u9891\uff1a{str(e)[:60]}")

        slide_tasks[task_id]["output_mp4"] = str(final_mp4)
        update_slide(f"\u5b8c\u6210\uff01\u5171 {len(page_images)} \u9875", "completed")
        slide_tasks[task_id]["_ts"] = time.time()

    except Exception as e:
        slide_tasks[task_id].update({"status": "error", "message": f"\u5931\u8d25\uff1a{str(e)}", "_ts": time.time()})


# ── Flask routes ───────────────────────────────────────────────────

_LOGIN_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>访问令牌</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;background:#0f1419;color:#e6e6e6;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}
form{background:#1a1f26;padding:32px;border-radius:10px;box-shadow:0 8px 32px rgba(0,0,0,.5);min-width:300px}
h1{margin:0 0 18px;font-size:17px;font-weight:500}
input{width:100%;padding:11px 12px;background:#252b33;border:1px solid #3a414b;color:#e6e6e6;border-radius:6px;font-size:14px;box-sizing:border-box;outline:none}
input:focus{border-color:#4a90e2}
button{width:100%;padding:11px;margin-top:14px;background:#4a90e2;border:0;color:#fff;border-radius:6px;font-size:14px;font-weight:500;cursor:pointer}
button:hover{background:#357abd}
.err{color:#ef6b6b;margin-top:10px;font-size:13px;min-height:18px;text-align:center}
</style></head>
<body><form method="POST" action="/login">
<h1>请输入访问令牌</h1>
<input type="password" name="token" autofocus required>
<button>进入</button>
<div class="err">__ERR__</div>
</form></body></html>"""

@app.before_request
def _auth_gate():
    if not AUTH_TOKEN:
        return
    if request.endpoint == "login" or request.path == "/login":
        return
    tok = request.args.get("token") or request.headers.get("X-Auth-Token") or request.cookies.get("auth_token")
    if tok and hmac.compare_digest(tok, AUTH_TOKEN):   # constant-time compare
        if request.args.get("token"):
            from urllib.parse import urlencode
            qs = {k: v for k, v in request.args.items() if k != "token"}
            target = request.path + (("?" + urlencode(qs)) if qs else "")
            resp = make_response(redirect(target))
            _secure = request.is_secure or request.headers.get("X-Forwarded-Proto") == "https"
            resp.set_cookie("auth_token", tok, max_age=30 * 24 * 3600, httponly=True, samesite="Lax", secure=_secure)
            return resp
        return
    if request.path.startswith("/api/"):
        return jsonify({"error": "unauthorized"}), 401
    return redirect("/login")

@app.route("/login", methods=["GET", "POST"])
def login():
    if not AUTH_TOKEN:
        return redirect("/")
    err = ""
    if request.method == "POST":
        tok = request.form.get("token", "")
        if tok and hmac.compare_digest(tok, AUTH_TOKEN):   # constant-time compare
            resp = make_response(redirect("/"))
            _secure = request.is_secure or request.headers.get("X-Forwarded-Proto") == "https"
            resp.set_cookie("auth_token", tok, max_age=30 * 24 * 3600, httponly=True, samesite="Lax", secure=_secure)
            return resp
        err = "令牌错误"
    return _LOGIN_HTML.replace("__ERR__", err)

@app.route("/")
def index():
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

@app.route("/api/gpu/status")
def api_gpu_status():
    try:
        import gpu_guard
        return jsonify(gpu_guard.get_status())
    except Exception as e:
        return jsonify({"enabled": False, "error": str(e)}), 500

@app.route("/api/check")
def api_check():
    return jsonify({
        "api_key":    bool(GOOGLE_API_KEY),
        "fish_key":   bool(FISH_API_KEY),
        "ffmpeg":     check_ffmpeg(),
        "google_tts": bool(GOOGLE_API_KEY),
        "librosa":     librosa_available(),
        "pymupdf":     pymupdf_available(),
        "pdf_support": pdf_support_available(),
    })

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if not GOOGLE_API_KEY:
        return jsonify({"error": "\u672a\u8bbe\u7f6e GOOGLE_API_KEY"}), 400
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "\u6ca1\u6709\u4e0a\u4f20\u6587\u4ef6"}), 400
    raw_model = request.args.get("ai_model", "gemini-3.5-flash")
    ai_model  = MODEL_MAP.get(raw_model, raw_model)
    transcribe_only = request.args.get("transcribe") == "1"
    transcribe_engine = request.args.get("engine", "gemini")
    job_id = uuid.uuid4().hex
    # Path().suffix strips any path components and never contains a slash, so a
    # hostile filename like "clip./a/b" can't make ext escape into a subpath
    # (which would 500 on file.save).
    ext = (Path(file.filename).suffix[1:].lower() or "mp4") if file.filename else "mp4"
    video_path = UPLOAD_DIR / f"{job_id}.{ext}"
    file.save(str(video_path))
    jobs[job_id] = {
        "status": "uploading_gemini", "progress": 0,
        "message": "\u5df2\u4e0a\u4f20\uff0c\u5f00\u59cb\u5904\u7406...",
        "filename": file.filename, "logs": [],
        "transcribe_only": transcribe_only,
        "transcribe_engine": transcribe_engine,
    }
    t = threading.Thread(target=generate_subtitles_task,
                         args=(job_id, video_path, ai_model),
                         kwargs={"transcribe_only": transcribe_only,
                                 "transcribe_engine": transcribe_engine}, daemon=True)
    t.start()
    return jsonify({"job_id": job_id})

@app.route("/api/status/<job_id>")
def api_status(job_id):
    def generate():
        for _ in range(_SSE_MAX_ITERS):
            job = get_or_load_job(job_id)
            if not job:
                yield f"data: {json.dumps({'error': '\u4efb\u52a1\u4e0d\u5b58\u5728', 'status': 'error'})}\n\n"
                break
            # Atomic snapshot: the worker thread inserts new keys into this same
            # job dict, and iterating it live raises "dictionary changed size
            # during iteration", killing the SSE stream. dict(job) copies in one
            # C-level op.
            data = dict(job)
            data.pop("_lock", None)
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            if job.get("status") in ("completed", "error"):
                break
            time.sleep(1)
    return Response(stream_with_context(generate()),
                    content_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/api/job_info/<job_id>")
def api_job_info(job_id):
    job = get_or_load_job(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    data = dict(job)   # atomic snapshot; worker may be inserting keys concurrently
    data.pop("_lock", None)
    return jsonify(data)

@app.route("/api/result/<job_id>")
def api_result(job_id):
    job = get_or_load_job(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    subtitles = job.get("subtitles", [])
    if not subtitles:
        sub_path = UPLOAD_DIR / f"{job_id}_subtitles.json"
        if sub_path.exists():
            with open(sub_path, encoding="utf-8") as f:
                subtitles = json.load(f)
    # Surface whether a dub already exists so the UI can reveal the
    # dub-download / 字幕+配音 burn buttons on refresh/restore WITHOUT forcing a
    # re-dub (you may only want to tweak subtitles and re-burn onto the old audio).
    dubbed_exists = (UPLOAD_DIR / f"{job_id}_dubbed.mp4").exists()
    dub_voice = ""
    vt = UPLOAD_DIR / f"{job_id}_audio" / "voice.txt"
    if dubbed_exists and vt.exists():
        try:
            dub_voice = vt.read_text().strip()
        except OSError:
            pass
    return jsonify({
        "subtitles": subtitles,
        "title": job.get("title", job.get("filename", job_id[:8])),
        "filename": job.get("filename", ""),
        "has_dub": dubbed_exists,
        "dub_voice": dub_voice,
    })

@app.route("/api/video/<job_id>")
def api_video(job_id):
    for ext in ("mp4", "mov", "avi", "mkv"):
        p = UPLOAD_DIR / f"{job_id}.{ext}"
        if p.exists():
            return send_file(str(p), mimetype="video/mp4", conditional=True)
    return jsonify({"error": "not found"}), 404

@app.route("/api/export/<job_id>")
def api_export_srt(job_id):
    job = get_or_load_job(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    subtitles = job.get("subtitles", [])
    if not subtitles:
        sub_path = UPLOAD_DIR / f"{job_id}_subtitles.json"
        if sub_path.exists():
            with open(sub_path, encoding="utf-8") as f:
                subtitles = json.load(f)
    def fmt_srt(s):
        h = int(s // 3600); m = int((s % 3600) // 60)
        sec = int(s % 60);  ms = int((s - int(s)) * 1000)
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"
    lines = []
    for i, sub in enumerate(subtitles, 1):
        lines += [str(i), f"{fmt_srt(sub['start'])} --> {fmt_srt(sub['end'])}",
                  strip_edge_punct(sub.get("translation", "")), ""]
    return Response("\n".join(lines), content_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename={job_id}.srt"})

@app.route("/api/update_subtitles/<job_id>", methods=["POST"])
def api_update_subtitles(job_id):
    """保存前端编辑后的字幕：覆写内存 job 与磁盘 JSON，烧录/导出随即生效。"""
    job = get_or_load_job(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    data = request.get_json(silent=True) or {}
    incoming = data.get("subtitles")
    if not isinstance(incoming, list):
        return jsonify({"error": "subtitles 必须是数组"}), 400
    clean = []
    for s in incoming:
        if not isinstance(s, dict):
            continue
        try:
            st = float(s.get("start"))
            en = float(s.get("end"))
        except (TypeError, ValueError):
            continue
        if en <= st:
            continue
        tr = str(s.get("translation", "")).strip()
        clean.append({
            "start": st, "end": en,
            "original": str(s.get("original", "")),
            "translation": tr,
        })
    if not clean:
        return jsonify({"error": "没有有效字幕"}), 400
    job["subtitles"] = clean
    sub_path = UPLOAD_DIR / f"{job_id}_subtitles.json"
    try:
        with open(sub_path, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
    except OSError as e:
        return jsonify({"error": f"写入失败: {e}"}), 500
    save_job_cache(job_id)
    return jsonify({"ok": True, "count": len(clean)})

_SRT_TS = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})")

def _srt_ts_to_sec(m):
    h, mm, s, ms = m
    return ((int(h) * 60 + int(mm)) * 60 + int(s)) + int(ms.ljust(3, "0")) / 1000.0

def parse_srt_text(text):
    """解析 SRT 文本为 [{start,end,original,translation}]，容忍前后噪声。导入时保持原文。"""
    subs = []
    for block in re.split(r"\n\s*\n", (text or "").strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        ti = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if ti is None:
            continue
        stamps = _SRT_TS.findall(lines[ti])
        if len(stamps) < 2:
            continue
        body = " ".join(lines[ti + 1:]).strip()
        if not body:
            continue
        st = _srt_ts_to_sec(stamps[0])
        en = _srt_ts_to_sec(stamps[1])
        if en <= st:
            continue
        subs.append({"start": st, "end": en, "original": body, "translation": body})
    return subs

@app.route("/api/import_srt/<job_id>", methods=["POST"])
def api_import_srt(job_id):
    """导入外部编辑好的 SRT，覆写该任务字幕（保持原文，不再规整）。"""
    job = get_or_load_job(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    srt_text = ""
    f = request.files.get("file")
    if f:
        srt_text = f.read().decode("utf-8-sig", "ignore")
    else:
        data = request.get_json(silent=True) or {}
        srt_text = data.get("srt", "")
    subs = parse_srt_text(srt_text)
    if not subs:
        return jsonify({"error": "未能从 SRT 解析到字幕"}), 400
    job["subtitles"] = subs
    sub_path = UPLOAD_DIR / f"{job_id}_subtitles.json"
    try:
        with open(sub_path, "w", encoding="utf-8") as fp:
            json.dump(subs, fp, ensure_ascii=False, indent=2)
    except OSError as e:
        return jsonify({"error": f"写入失败: {e}"}), 500
    save_job_cache(job_id)
    return jsonify({"ok": True, "count": len(subs), "subtitles": subs})

@app.route("/api/download/<job_id>", methods=["GET", "POST"])
def api_download(job_id):
    burn = request.args.get("burn") == "1" or request.method == "POST"
    if burn:
        data = request.get_json(silent=True) or {}
        job = get_or_load_job(job_id)
        if not job:
            return jsonify({"error": "not found"}), 404
        subtitles = job.get("subtitles", [])
        if not subtitles:
            sub_path = UPLOAD_DIR / f"{job_id}_subtitles.json"
            if sub_path.exists():
                with open(sub_path, encoding="utf-8") as f:
                    subtitles = json.load(f)
        video_path = None
        for ext in ("mp4", "mov", "avi", "mkv"):
            p = UPLOAD_DIR / f"{job_id}.{ext}"
            if p.exists():
                video_path = p
                break
        if not video_path:
            return jsonify({"error": "video not found"}), 404
        # 结尾裁剪（如 NotebookLM 广告尾页）：限制输出时长，并丢弃落在被裁区间的字幕。
        trim_target = None
        try:
            trim_end = float(data.get("trim_end", 0) or 0)
        except (TypeError, ValueError):
            trim_end = 0
        if trim_end > 0:
            dur = get_video_duration(video_path)
            if dur and dur > trim_end + 0.5:
                trim_target = dur - trim_end
                subtitles = [{**s, "end": min(s["end"], trim_target)}
                             for s in subtitles if s["start"] < trim_target - 0.05]
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "v:0", str(video_path)],
            capture_output=True, text=True
        )
        try:
            info = json.loads(r.stdout or "{}")
        except json.JSONDecodeError:
            info = {}
        streams = info.get("streams") or [{}]   # empty list (audio-only file) -> safe default
        s0 = streams[0] if streams else {}
        video_w = int(s0.get("width", 1920))
        video_h = int(s0.get("height", 1080))
        ass_content = build_ass(
            subtitles, video_w, video_h,
            font_size_frac=data.get("font_size_frac", 0.04),
            max_chars=data.get("max_chars", 36),
            outline=data.get("outline", 3),
            sub_color=data.get("sub_color", "&H0000FFFF"),
            x_frac=data.get("x_frac", 0.5),
            y_frac=data.get("y_frac", 0.92),
            ass_align=data.get("ass_align", 2),
        )
        ass_path = UPLOAD_DIR / f"{job_id}_burn.ass"
        ass_path.write_text(ass_content, encoding="utf-8")
        output_path = UPLOAD_DIR / f"{job_id}_burned.mp4"
        nvenc = nvenc_enabled()
        vcodec = "h264_nvenc" if nvenc else "libx264"
        def run_sub_burn(vc):
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vf", f"ass={escape_ass_path(ass_path)}",
                "-c:v", vc, "-c:a", "copy",
            ]
            if trim_target is not None:
                cmd += ["-t", f"{trim_target:.3f}"]   # 裁掉结尾广告尾页
            cmd += [str(output_path)]
            subprocess.run(cmd, check=True, capture_output=True)
        if output_path.exists():
            output_path.unlink()   # don't serve a stale burn if this run fails
        try:
            run_sub_burn(vcodec)
        except subprocess.CalledProcessError:
            if vcodec == "libx264":
                raise          # no codec to fall back to — surface the failure
            run_sub_burn("libx264")
        return send_file(str(output_path), as_attachment=True,
                         download_name="video_with_subtitles.mp4")
    else:
        for ext in ("mp4", "mov", "avi", "mkv"):
            p = UPLOAD_DIR / f"{job_id}.{ext}"
            if p.exists():
                return send_file(str(p), as_attachment=True)
        return jsonify({"error": "not found"}), 404

@app.route("/api/slide/clear_clips/<job_id>", methods=["POST"])
def api_slide_clear_clips(job_id):
    """清除 Slide 视频片段缓存（保留旁白文字和音频），方便重新渲染。"""
    work_dir = UPLOAD_DIR / f"{job_id}_slides"
    if not work_dir.exists():
        return jsonify({"error": "slide工作目录不存在"}), 404
    deleted = []
    for f in work_dir.glob("clip_*.mp4"):
        f.unlink()
        deleted.append(f.name)
    # Also remove merged output so it gets regenerated
    for fname in ["output.mp4", "clips_list.txt"]:
        p = work_dir / fname
        if p.exists():
            p.unlink()
    return jsonify({"ok": True, "deleted": len(deleted), "files": deleted})

@app.route("/api/reprocess/<job_id>", methods=["POST"])
def api_reprocess(job_id):
    job = get_or_load_job(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    raw_model = request.args.get("ai_model", "gemini-3.5-flash")
    ai_model  = MODEL_MAP.get(raw_model, raw_model)
    # Preserve the original mode unless the query explicitly overrides it.
    if request.args.get("transcribe") is not None:
        transcribe_only = request.args.get("transcribe") == "1"
    else:
        transcribe_only = bool(job.get("transcribe_only"))
    transcribe_engine = request.args.get("engine") or job.get("transcribe_engine", "gemini")
    video_path = None
    for ext in ("mp4", "mov", "avi", "mkv"):
        p = UPLOAD_DIR / f"{job_id}.{ext}"
        if p.exists():
            video_path = p
            break
    if not video_path:
        return jsonify({"error": "video not found"}), 404
    job.update({"status": "uploading_gemini", "progress": 0,
                "message": "\u91cd\u65b0\u5904\u7406\u4e2d...", "logs": [], "subtitles": [],
                "transcribe_only": transcribe_only,
                "transcribe_engine": transcribe_engine})
    t = threading.Thread(target=generate_subtitles_task,
                         args=(job_id, video_path, ai_model),
                         kwargs={"transcribe_only": transcribe_only,
                                 "transcribe_engine": transcribe_engine}, daemon=True)
    t.start()
    return jsonify({"ok": True})

@app.route("/api/dub/<job_id>", methods=["POST"])
def api_dub(job_id):
    job = get_or_load_job(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    data      = request.get_json(silent=True) or {}
    voice     = data.get("voice", "edge_yunjian")
    tts_model = data.get("tts_model", "gemini-3.1-flash-live-preview-tts")
    t = threading.Thread(target=dubbing_task, args=(job_id, voice, tts_model), daemon=True)
    t.start()
    return jsonify({"ok": True})

@app.route("/api/dub_status/<job_id>")
def api_dub_status(job_id):
    def generate():
        for _ in range(_SSE_MAX_ITERS):
            data = dict(dub_tasks.get(job_id, {"status": "pending", "progress": 0, "message": "\u7b49\u5f85\u4e2d..."}))
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            if data.get("status") in ("completed", "error"):
                break
            time.sleep(1)
    return Response(stream_with_context(generate()),
                    content_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/api/dub_download/<job_id>")
def api_dub_download(job_id):
    path = UPLOAD_DIR / f"{job_id}_dubbed.mp4"
    if path.exists():
        return send_file(str(path), as_attachment=True, download_name="dubbed_video.mp4")
    return jsonify({"error": "not found"}), 404

@app.route("/api/burn_dub_start/<job_id>", methods=["POST"])
def api_burn_dub_start(job_id):
    data = request.get_json(silent=True) or {}
    task_id = uuid.uuid4().hex
    t = threading.Thread(
        target=burn_dub_task,
        args=(task_id, job_id,
              data.get("voice", "edge_yunjian"),
              data.get("x_frac", 0.5), data.get("y_frac", 0.92),
              data.get("font_size_frac", 0.04), data.get("ass_align", 2),
              data.get("outline", 3), data.get("sub_color", "&H0000FFFF"),
              data.get("max_chars", 36)),
        daemon=True
    )
    t.start()
    return jsonify({"task_id": task_id})

@app.route("/api/burn_dub_status/<task_id>")
def api_burn_dub_status(task_id):
    def generate():
        for _ in range(_SSE_MAX_ITERS):
            data = dict(burn_tasks.get(task_id, {"status": "pending", "message": "\u7b49\u5f85\u4e2d..."}))
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            if data.get("status") in ("completed", "error"):
                break
            time.sleep(1)
    return Response(stream_with_context(generate()),
                    content_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/api/burn_dub_download/<task_id>")
def api_burn_dub_download(task_id):
    task = burn_tasks.get(task_id, {})
    output = task.get("output")
    if output and Path(output).exists():
        return send_file(output, as_attachment=True, download_name="dubbed_with_subtitles.mp4")
    return jsonify({"error": "not found"}), 404

@app.route("/api/slide_dub_start/<job_id>", methods=["POST"])
def api_slide_dub_start(job_id):
    file = request.files.get("file")
    voice = request.form.get("voice", "edge_yunjian")
    tts_model = request.form.get("tts_model", "gemini-3.1-flash-live-preview-tts")
    burn_subs = request.form.get("burn_subs", "1") != "0"

    def _f(name, default):
        try:
            return float(request.form.get(name, default))
        except (TypeError, ValueError):
            return default
    sub_style = {
        "font_size_frac": _f("font_size_frac", 0.045),
        "sub_color": request.form.get("sub_color", "&H0000FFFF"),
        "y_frac": _f("y_frac", 0.92),
        "ass_align": int(_f("ass_align", 2)),
    }
    task_id = uuid.uuid4().hex

    if file:
        suffix = Path(file.filename).suffix if file.filename else ".pdf"
        save_path = UPLOAD_DIR / f"{task_id}_slides_input{suffix}"
        file.save(str(save_path))
    else:
        # No file uploaded — reuse existing input.pdf from previous run
        work_dir = UPLOAD_DIR / f"{job_id}_slides"
        existing_pdf = work_dir / "input.pdf" if work_dir.exists() else None
        if existing_pdf and existing_pdf.exists():
            # Copy to temp path so slide_dub_task doesn't copy2 same→same
            import shutil as _sh
            save_path = UPLOAD_DIR / f"{task_id}_slides_reuse.pdf"
            _sh.copy2(existing_pdf, save_path)
        else:
            return jsonify({"error": "no file"}), 400
    t = threading.Thread(target=slide_dub_task, args=(task_id, job_id, str(save_path), voice, tts_model, burn_subs, sub_style), daemon=True)
    t.start()
    return jsonify({"task_id": task_id})

@app.route("/api/slide_dub_status/<task_id>")
def api_slide_dub_status(task_id):
    def generate():
        for _ in range(_SSE_MAX_ITERS):
            data = dict(slide_tasks.get(task_id, {"status": "pending", "message": "\u7b49\u5f85\u4e2d..."}))
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            if data.get("status") in ("completed", "error"):
                break
            time.sleep(1)
    return Response(stream_with_context(generate()),
                    content_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/api/slide_dub_download/<task_id>")
def api_slide_dub_download(task_id):
    task = slide_tasks.get(task_id, {})
    output = task.get("output_mp3")
    if output and Path(output).exists():
        return send_file(output, as_attachment=True, download_name="slide_narration.mp3")
    return jsonify({"error": "not found"}), 404

@app.route("/api/slide_dub_download_mp4/<task_id>")
def api_slide_dub_download_mp4(task_id):
    task = slide_tasks.get(task_id, {})
    output = task.get("output_mp4")
    if output and Path(output).exists():
        return send_file(output, as_attachment=True, download_name="slide_narration.mp4")
    return jsonify({"error": "not found"}), 404


@app.route("/api/slide_dub_latest")
def api_slide_dub_latest():
    """Recover the most recent slide-dubbing task (optionally for one job_id) so the
    frontend can re-attach progress / download buttons after a page refresh — the
    task lives in a background thread, but the browser's task_id was lost on reload."""
    job_id = request.args.get("job_id")
    best, best_ts = None, -1.0
    for tid, t in slide_tasks.items():
        if job_id and t.get("job_id") != job_id:
            continue
        ts = float(t.get("_ts") or 0)
        if ts >= best_ts:   # ties → later insertion order wins (dict is ordered)
            best_ts, best = ts, (tid, t)
    if not best:
        return jsonify({"task": None})
    tid, t = best
    return jsonify({"task": {
        "task_id": tid,
        "status": t.get("status"),
        "message": t.get("message", ""),
        "job_id": t.get("job_id"),
        "age_sec": time.time() - best_ts if best_ts > 0 else None,
        "has_mp3": bool(t.get("output_mp3") and Path(t["output_mp3"]).exists()),
        "has_mp4": bool(t.get("output_mp4") and Path(t["output_mp4"]).exists()),
    }})


def detect_gpu():
    """Detect GPU and nvenc availability, print startup hint."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            gpu_name, driver_ver = [x.strip() for x in r.stdout.strip().split(",", 1)]
            # Check nvenc: requires driver >= 570.0
            try:
                major = int(driver_ver.split(".")[0])
                nvenc_ok = major >= 570
            except Exception:
                nvenc_ok = False
            if not ALLOW_NVENC:
                print(f"   GPU:            {gpu_name} (driver {driver_ver}) \u26d4 nvenc \u5df2\u88ab\u7b56\u7565\u7981\u7528(ALLOW_NVENC=0)\uff0c\u5168\u7a0b\u7528 libx264")
            elif nvenc_ok:
                print(f"   GPU:            {gpu_name} (driver {driver_ver}) \u2705 nvenc \u53ef\u7528")
            else:
                print(f"   GPU:            {gpu_name} (driver {driver_ver}) \u26a0\ufe0f nvenc \u9700\u9a71\u52a8 \u2265570.0\uff0c\u5f53\u524d\u56de\u9000 libx264")
        else:
            print("   GPU:            \u672a\u68c0\u6d4b\u5230 NVIDIA GPU\uff0c\u4f7f\u7528 CPU \u7f16\u7801")
    except FileNotFoundError:
        print("   GPU:            nvidia-smi \u4e0d\u5728\uff0c\u4f7f\u7528 CPU \u7f16\u7801")
    except Exception as e:
        print(f"   GPU:            \u68c0\u6d4b\u5931\u8d25 ({e})\uff0c\u4f7f\u7528 CPU \u7f16\u7801")

def _purge_old_tasks():
    """Periodically purge completed tasks older than _TASK_TTL to prevent memory leak."""
    while True:
        time.sleep(300)  # check every 5 minutes
        now = time.time()
        for store in (dub_tasks, burn_tasks, slide_tasks):
            try:
                # snapshot with list(...): a request thread inserting a new task
                # key mid-iteration would otherwise raise "dictionary changed
                # size during iteration" and kill this daemon, stopping ALL
                # future purging (unbounded growth). Also guard the body so a
                # transient error never terminates the thread.
                to_del = [k for k, v in list(store.items())
                          if v.get("status") in ("completed", "error")
                          and v.get("_ts", now) < now - _TASK_TTL]
                for k in to_del:
                    store.pop(k, None)
            except Exception as e:
                print(f"[purge] error: {e}", flush=True)


# ─── MTV Routes ───────────────────────────────────────────────────────────────

MTV_VISUAL_EXTS = MTV_VIDEO_EXTS | MTV_IMAGE_EXTS | ALLOWED_PDF_EXTS


@app.route('/api/mtv/create', methods=['POST'])
def api_mtv_create():
    """Start an MTV generation job.
    Form fields: music (file), visuals[] (files), beats_per_switch (int), resolution (str).
    """
    if not check_ffmpeg():
        return jsonify({'error': 'ffmpeg 不可用，无法生成 MTV'}), 500

    music_file = request.files.get('music')
    if not music_file or not music_file.filename:
        return jsonify({'error': '请上传音乐文件'}), 400

    music_ext = Path(music_file.filename).suffix.lower()
    if music_ext not in ALLOWED_MUSIC_EXTS:
        return jsonify({'error': f'不支持的音乐格式 {music_ext}'}), 400

    visual_files = request.files.getlist('visuals[]')
    if not visual_files:
        return jsonify({'error': '请上传至少一个视觉素材文件'}), 400

    try:
        beats_per_switch = int(request.form.get('beats_per_switch', 1))
    except ValueError:
        beats_per_switch = 1
    if beats_per_switch not in (1, 2, 4, 8):
        beats_per_switch = 1

    resolution = request.form.get('resolution', '720p')
    if resolution not in ('720p', '1080p'):
        resolution = '720p'

    orientation = request.form.get('orientation', 'landscape')
    if orientation not in ('landscape', 'portrait'):
        orientation = 'landscape'

    job_id = str(uuid.uuid4())
    work_dir = os.path.join(str(UPLOAD_DIR), 'mtv', job_id)
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    # Save music
    music_path = os.path.join(work_dir, f'music{music_ext}')
    music_file.save(music_path)

    # Save visual files and classify
    source_infos = []
    for vf in visual_files:
        if not vf or not vf.filename:
            continue
        ext = Path(vf.filename).suffix.lower()
        if ext not in MTV_VISUAL_EXTS:
            continue
        safe_name = str(uuid.uuid4()) + ext
        save_path = os.path.join(work_dir, safe_name)
        vf.save(save_path)
        if ext in ALLOWED_PDF_EXTS:
            src_type = 'pdf'
        elif ext in MTV_IMAGE_EXTS:
            src_type = 'image'
        else:
            src_type = 'video'
        source_infos.append({'type': src_type, 'path': save_path})

    if not source_infos:
        return jsonify({'error': '没有有效的视觉素材文件（支持 MP4/MOV/AVI/MKV/PDF/JPG/PNG 等）'}), 400

    with mtv_jobs_lock:
        mtv_jobs[job_id] = {
            'status': 'queued', 'progress': 0,
            'message': '准备开始...', 'output_path': None,
        }

    def _update(progress: int, status: str, message: str, output_path: str = None):
        with mtv_jobs_lock:
            if job_id in mtv_jobs:
                mtv_jobs[job_id].update({
                    'progress': progress,
                    'status': status,
                    'message': message,
                })
                if output_path:
                    mtv_jobs[job_id]['output_path'] = output_path

    threading.Thread(
        target=process_mtv_job,
        args=(job_id, source_infos, music_path,
              beats_per_switch, resolution, work_dir, _update, orientation),
        daemon=True,
    ).start()

    return jsonify({'job_id': job_id})


@app.route('/api/mtv/status/<job_id>')
def api_mtv_status(job_id: str):
    """SSE stream for MTV job progress."""
    def generate():
        last_hash = None
        deadline = time.time() + 3600
        while time.time() < deadline:
            with mtv_jobs_lock:
                if job_id not in mtv_jobs:
                    yield f"data: {json.dumps({'error': 'MTV 任务不存在'})}\n\n"
                    return
                job_data = {k: v for k, v in mtv_jobs[job_id].items()
                            if k != 'output_path'}

            s = json.dumps(job_data, ensure_ascii=False)
            h = hash(s)
            if h != last_hash:
                yield f"data: {s}\n\n"
                last_hash = h

            if job_data['status'] in ('completed', 'error'):
                return
            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
        'Connection': 'keep-alive',
    })


@app.route('/api/mtv/download/<job_id>')
def api_mtv_download(job_id: str):
    """Download the finished MTV file."""
    with mtv_jobs_lock:
        job = dict(mtv_jobs.get(job_id, {}))
    if job.get('status') != 'completed':
        return jsonify({'error': 'MTV 尚未完成'}), 400
    out = job.get('output_path', '')
    if not out or not os.path.exists(out):
        return jsonify({'error': '输出文件不存在'}), 404
    return send_file(out, as_attachment=True,
                     download_name='mtv_output.mp4', mimetype='video/mp4')


# ── 卡拉OK字幕(B 强制对齐;子进程调 infinitetalk 环境的 align_karaoke_b.py) ──────
karaoke_tasks = {}
_KARA_PY = "/home/user/miniconda3/envs/infinitetalk/bin/python"
_KARA_SCRIPT = str(Path(__file__).resolve().parent / "align_karaoke_b.py")
_TC_FONT_SERIF = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
_TC_FONT_SANS = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
_SUNO_SCRIPT = str(Path(__file__).resolve().parent / "suno_karaoke.py")
_SUNO_API = os.environ.get("SUNO_API", "http://localhost:3000")


def _suno_find_song(q):
    """q 是 song_id(uuid)→直接用;否则按歌名在 Suno 曲库搜标题,返回 (song_id, title)。需 suno-api(:3000)在跑。"""
    q = (q or "").strip()
    if re.match(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-", q):
        return q, ""
    import urllib.request
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # 本地调用绕过 SOCKS5 代理
    ql = q.lower()
    for page in range(0, 25):   # /api/get 每页~20首,翻页搜全库(最多~500首)
        try:
            data = json.loads(op.open(f"{_SUNO_API}/api/get?page={page}", timeout=60).read())
        except Exception:
            break
        if not data:
            break
        for x in data:
            if ql in (x.get("title") or "").lower():
                return x.get("id"), x.get("title")
    raise RuntimeError(f"Suno 曲库(翻页搜完)没找到标题含「{q}」的歌(确认歌名对、suno-api 在跑)")


def _burn_titlecard(src, dst, tc):
    """成片开头烧歌名(第1行)+词曲作者(第2行)字幕卡,均居中。
    淡入≈1.5s → 保持 → 结束前2s淡出 → dur秒后消失(落前奏里)。textfile= 免转义。"""
    import tempfile
    font = _TC_FONT_SANS if tc.get("font") == "sans" else _TC_FONT_SERIF
    wh = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=width,height", "-of", "csv=p=0", str(src)],
                        capture_output=True, text=True).stdout.strip()
    W, H = (list(map(int, wh.split(",")[:2])) if "," in wh else [960, 540])
    size = int(tc.get("size", 50)); csize = max(16, int(size * 0.5))
    color = (tc.get("color", "#ffffff") or "#ffffff").lstrip("#")
    y = H * int(tc.get("ypos", 36)) / 100.0
    dur = float(tc.get("dur", 10)); fo = max(2.0, dur - 2.0)
    A = (f"if(lt(t\\,1.5)\\,t/1.5\\,if(lt(t\\,{fo:.2f})\\,1\\,"
         f"if(lt(t\\,{dur:.2f})\\,({dur:.2f}-t)/({dur:.2f}-{fo:.2f})\\,0)))")
    tmps, filters = [], []

    def _df(text, fs, bw, yy):
        tf = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        tf.write(text); tf.close(); tmps.append(tf.name)
        return (f"drawtext=fontfile={font}:textfile={tf.name}:fontcolor=0x{color}:"
                f"fontsize={fs}:borderw={bw}:bordercolor=black@0.7:"
                f"x=(w-text_w)/2:y={yy:.0f}:alpha={A}")

    filters.append(_df(tc.get("title", ""), size, 2, y))
    cr = (tc.get("credits") or "").strip()
    if cr:
        filters.append(_df(cr, csize, 1.5, y + size + 24))
    r = subprocess.run(["ffmpeg", "-y", "-i", str(src), "-vf", ",".join(filters),
                        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                        "-c:a", "copy", str(dst)], capture_output=True, text=True)
    for f in tmps:
        try:
            os.remove(f)
        except OSError:
            pass
    if r.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError((r.stderr or "")[-200:])


def karaoke_task(task_id, video_path, lyrics_path, out_path, p):
    suno = p.get("mode") == "suno"
    karaoke_tasks[task_id] = {"status": "running",
        "message": ("Suno 强化对齐 + 烧录中…" if suno else "强制对齐 + 烧录中(约1-2分钟)…"), "_ts": time.time()}
    try:
        if suno:    # Suno 逐词对齐(准,英文也行);lyrics 文件可选=双语译文(第二行中文)
            cmd = [_KARA_PY, _SUNO_SCRIPT, "--video", video_path, "--song-id", p["song_id"],
                   "--out", out_path, "--font-size", str(p.get("font_size", 50)),
                   "--margin-v", str(p.get("margin_v", 60)), "--outline", str(p.get("outline", 2.5)),
                   "--primary", p.get("primary", "#33E0FF"), "--secondary", p.get("secondary", "#FFFFFF"),
                   "--lead-in", str(p.get("lead_in", 0.6)), "--linger", str(p.get("linger", 1.2))]
            if p.get("bilingual"):
                cmd += ["--trans", lyrics_path]
        else:
            cmd = [_KARA_PY, _KARA_SCRIPT, "--video", video_path, "--lyrics", lyrics_path,
                   "--out", out_path, "--device", p.get("device", "cuda"), "--lang", p.get("lang", "zh"),
                   "--font-size", str(p.get("font_size", 54)), "--margin-v", str(p.get("margin_v", 50)),
                   "--outline", str(p.get("outline", 3.0)), "--primary", p.get("primary", "#33E0FF"),
                   "--secondary", p.get("secondary", "#FFFFFF"), "--lead-in", str(p.get("lead_in", 0.7)),
                   "--linger", str(p.get("linger", 1.5))]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError((r.stderr or r.stdout or "")[-400:])
        tc = p.get("titlecard")
        if tc and tc.get("title"):
            karaoke_tasks[task_id].update({"message": "烧录开头标题卡…", "_ts": time.time()})
            tmp = out_path + ".pretitle.mp4"
            os.replace(out_path, tmp)
            try:
                _burn_titlecard(tmp, out_path, tc)
                os.remove(tmp)
            except Exception as e:
                os.replace(tmp, out_path)            # 失败保留无卡版
                print("⚠ 标题卡烧录失败,保留无卡版:", e)
        karaoke_tasks[task_id].update({"status": "done", "message": "完成 ✓", "output": out_path, "_ts": time.time()})
    except Exception as e:
        karaoke_tasks[task_id].update({"status": "error", "message": str(e)[:300], "_ts": time.time()})


@app.route("/karaoke")
def karaoke_page():
    return make_response(render_template("karaoke.html"))


@app.route("/api/karaoke/start", methods=["POST"])
def api_karaoke_start():
    video = request.files.get("video")
    lyrics = (request.form.get("lyrics") or "").strip()
    mode = (request.form.get("mode") or "force").strip()
    if not video or not video.filename:
        return jsonify({"error": "需要上传视频"}), 400
    if mode != "suno" and not lyrics:
        return jsonify({"error": "需要上传视频并粘贴歌词"}), 400
    task_id = uuid.uuid4().hex
    suf = Path(video.filename).suffix or ".mp4"
    vpath = UPLOAD_DIR / f"kara_{task_id}{suf}"
    lpath = UPLOAD_DIR / f"kara_{task_id}.txt"
    opath = UPLOAD_DIR / f"kara_{task_id}_out.mp4"
    video.save(str(vpath))
    if lyrics:
        lpath.write_text(lyrics, encoding="utf-8")    # Suno 模式:此处=双语译文(可选,做第二行中文)
    g = request.form.get
    if mode == "suno":
        try:
            sid, _stitle = _suno_find_song(g("song") or "")
        except Exception as e:
            return jsonify({"error": str(e)[:200]}), 400
    p = {"device": g("device", "cuda"), "lang": g("lang", "zh"),
         "font_size": g("font_size", "54"), "margin_v": g("margin_v", "50"),
         "outline": g("outline", "3.0"), "primary": g("primary", "#33E0FF"),
         "secondary": g("secondary", "#FFFFFF"), "lead_in": g("lead_in", "0.7"),
         "linger": g("linger", "1.5")}
    if mode == "suno":
        p["mode"] = "suno"; p["song_id"] = sid; p["bilingual"] = bool(lyrics)
    # 开头字幕卡(歌名/词曲作者 + 样式);歌名留空 = 不加卡
    _st = (g("song_title") or "").strip()
    p["titlecard"] = ({"title": _st, "credits": (g("song_credits") or "").strip(),
                       "font": g("tc_font", "serif"), "size": int(float(g("tc_size", "50"))),
                       "color": g("tc_color", "#ffffff"), "ypos": int(float(g("tc_ypos", "36"))),
                       "dur": float(g("tc_dur", "10"))} if _st else None)
    threading.Thread(target=karaoke_task, args=(task_id, str(vpath), str(lpath), str(opath), p), daemon=True).start()
    return jsonify({"task_id": task_id})


@app.route("/api/karaoke/status/<task_id>")
def api_karaoke_status(task_id):
    t = karaoke_tasks.get(task_id, {"status": "pending", "message": "等待中…"})
    return jsonify({"status": t.get("status"), "message": t.get("message"), "ready": t.get("status") == "done"})


@app.route("/api/karaoke/download/<task_id>")
def api_karaoke_download(task_id):
    out = (karaoke_tasks.get(task_id) or {}).get("output")
    if not out or not os.path.exists(out):
        return ("", 404)
    return send_file(out, mimetype="video/mp4",
                     as_attachment=(request.args.get("download") == "1"),
                     download_name="karaoke.mp4")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 28500))
    print(f"Video Subtitle Generator starting on http://0.0.0.0:{port}")
    print(f"   GOOGLE_API_KEY: {'set' if GOOGLE_API_KEY else 'NOT SET'}")
    print(f"   Gemini keys:    {len(GEMINI_KEYS)} 个 key 轮询" + (f" (from {Path(GEMINI_KEYS_FILE).name})" if len(GEMINI_KEYS) > 1 else " (单 key)"))
    print(f"   FISH_API_KEY:   {'set' if FISH_API_KEY else 'not set'}")
    detect_gpu()

    # Optional: prewarm Whisper model in background so the first user request
    # doesn't trigger a 10-minute model download with no UI feedback.
    # Set WHISPER_PREWARM=1 to enable.
    if WHISPER_ENABLED and os.environ.get("WHISPER_PREWARM", "0") == "1":
        def _prewarm():
            print(f"   Whisper:        preloading {WHISPER_MODEL_SIZE} on {WHISPER_DEVICE}...")
            m = get_whisper_model()
            if m is not None:
                print(f"   Whisper:        ready ({WHISPER_MODEL_SIZE} on {WHISPER_DEVICE})")
            else:
                print(f"   Whisper:        preload failed, will fall back to Gemini at request time")
        threading.Thread(target=_prewarm, daemon=True).start()

    # Start background task purger
    threading.Thread(target=_purge_old_tasks, daemon=True).start()

    # Start the RTX 3080 thermal shield (continuous temp monitor + emergency
    # cutoff). The pre-job gate in _ollama_translate works regardless, but the
    # monitor is what catches a runaway mid-inference.
    try:
        import gpu_guard
        gpu_guard.start_monitor()
    except Exception as _e:
        print(f"   GPU guard:      failed to start ({_e})")

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
