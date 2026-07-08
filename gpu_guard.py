"""gpu_guard.py — RTX 3080 thermal shield for GPU jobs (Ollama / Whisper).

This box's RTX 3080 has a recurring thermal / PCIe drop-off problem ("GPU is
lost"). Sustained GPU load — gemma4 translation via Ollama, or Whisper — can
push the card hot and, left unchecked, crash it. This module is the software
shield. Three layers:

  1. wait_until_cool() — a BLOCKING gate. Call it right before launching a GPU
     job. If the card is at/above TRIP_C and stays there, it blocks until the
     card cools to RESUME_C, then returns. Stops us starting a job on a card
     that's already hot from the previous one.

  2. GpuMonitor — a daemon thread that samples temperature continuously, tracks
     peak + a rolling hot/normal state, logs threshold crossings, and serves
     status() (used by /api/gpu/status and gpu_watch.py).

  3. Emergency cutoff — within a single long inference the gate can't help
     (one Ollama call can run minutes). So a job registers an "abort" callback
     (e.g. closing its in-flight HTTP client) via register_abortable(); if the
     monitor sees temp cross HARD_CEIL_C it fires every abort callback, killing
     the in-flight request. The caller then falls back to Gemini.

Hysteresis (trip 65 / release 55) avoids flapping at the line. A sustained
check (N consecutive hot reads) avoids stalling the pipeline on a momentary
spike. Everything is env-tunable; defaults match the user's stated thresholds
(>=65 wait, resume at 55, ~90s is normal).

nvidia-smi is a local binary — no network, so ALL_PROXY in .env doesn't affect
it. If nvidia-smi is missing or the card has dropped off the bus, temp reads
return None and the gate declines to block (the job will fail/fallback on its
own rather than hang forever waiting on a dead sensor).
"""

import os
import sys
import time
import shutil
import threading
import subprocess


def _envf(name, default):
    try:
        return float(os.environ.get(name, default))
    except (ValueError, TypeError):
        return float(default)


def _envi(name, default):
    try:
        return int(os.environ.get(name, default))
    except (ValueError, TypeError):
        return int(default)


ENABLED      = os.environ.get("GPU_GUARD_ENABLED", "1") != "0"
TRIP_C       = _envf("GPU_TRIP_C", 65)        # block new jobs at/above this
RESUME_C     = _envf("GPU_RESUME_C", 55)      # release once cooled to this
HARD_CEIL_C  = _envf("GPU_HARD_CEIL_C", 88)   # emergency: abort in-flight jobs
SAMPLE_SEC   = _envf("GPU_SAMPLE_SEC", 3.0)   # monitor + gate sample interval
TRIP_SUSTAIN = _envi("GPU_TRIP_SUSTAIN", 2)   # consecutive hot reads to trip gate
SLOW_WARN_S  = _envf("GPU_COOL_SLOW_WARN_S", 90)   # warn if cooldown drags past this
COOL_CAP_S   = _envf("GPU_COOL_TIMEOUT_S", 300)    # hard cap: proceed-with-warning

def _resolve_smi():
    # PATH-based first. But under WSL the binary lives in /usr/lib/wsl/lib,
    # which is on interactive shells' PATH yet NOT on the daemon PATH that
    # start.sh inherits (/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:
    # /snap/bin) — so when launched as a service shutil.which() misses it and
    # temp reads silently return None ("nogpu"). Fall back to known locations.
    p = shutil.which("nvidia-smi")
    if p:
        return p
    for cand in ("/usr/lib/wsl/lib/nvidia-smi",   # WSL GPU passthrough
                 "/usr/bin/nvidia-smi",
                 "/usr/local/bin/nvidia-smi"):
        if os.path.exists(cand):
            return cand
    return "nvidia-smi"


_SMI = _resolve_smi()


def _log(msg, log_fn=None):
    line = f"[gpu-guard] {msg}"
    print(line, file=sys.stderr, flush=True)
    if log_fn:
        try:
            log_fn(msg)
        except Exception:
            pass


def gpu_temp():
    """Current GPU temperature in °C (max across GPUs), or None if unreadable
    (nvidia-smi missing, non-zero exit, '[N/A]', or card dropped off the bus)."""
    try:
        out = subprocess.run(
            [_SMI, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        temps = []
        for ln in (out.stdout or "").splitlines():
            ln = ln.strip()
            if ln.isdigit():
                temps.append(int(ln))
        return max(temps) if temps else None
    except Exception:
        return None


# ---- emergency abort registry --------------------------------------------
_abort_lock = threading.Lock()
_abortables = set()


def register_abortable(closer):
    """Register a zero-arg callable the monitor will invoke if temp crosses
    HARD_CEIL_C (e.g. an httpx client's .close to kill an in-flight request)."""
    with _abort_lock:
        _abortables.add(closer)


def unregister_abortable(closer):
    with _abort_lock:
        _abortables.discard(closer)


def _fire_aborts():
    with _abort_lock:
        closers = list(_abortables)
    for c in closers:
        try:
            c()
        except Exception:
            pass


# ---- blocking pre-job gate -----------------------------------------------
def wait_until_cool(log_fn=None, reason=""):
    """BLOCK until the GPU is cool enough to start a job.

    Returns immediately if the guard is disabled, the temp is unreadable, or
    the card is below TRIP_C. Otherwise (sustained >= TRIP_C) it blocks until
    the card cools to <= RESUME_C, then returns. Caps the wait at COOL_CAP_S
    and proceeds with a loud warning rather than hanging the pipeline forever.
    """
    if not ENABLED:
        return
    tag = f"（{reason}）" if reason else ""

    # Sustained-trip check: require TRIP_SUSTAIN consecutive hot reads so a
    # momentary spike at job start doesn't stall us.
    hot_reads = 0
    for _ in range(max(1, TRIP_SUSTAIN)):
        t = gpu_temp()
        if t is None:
            _log(f"读不到 GPU 温度（nvidia-smi 不可用/掉卡？），跳过温控{tag}", log_fn)
            return
        if t >= TRIP_C:
            hot_reads += 1
        else:
            return  # below trip — go
        if hot_reads < TRIP_SUSTAIN:
            time.sleep(min(SAMPLE_SEC, 2.0))
    # confirmed hot
    start = time.monotonic()
    last_note = 0.0
    warned_slow = False
    _log(f"GPU {t}°C ≥ {TRIP_C:.0f}°C，暂停{tag}，等待冷却到 {RESUME_C:.0f}°C…", log_fn)
    while True:
        t = gpu_temp()
        if t is None:
            _log(f"冷却等待中读不到温度，放行{tag}", log_fn)
            return
        if t <= RESUME_C:
            waited = time.monotonic() - start
            _log(f"GPU 已降到 {t}°C（等了 {waited:.0f}s），继续{tag}", log_fn)
            return
        elapsed = time.monotonic() - start
        if not warned_slow and elapsed >= SLOW_WARN_S:
            warned_slow = True
            _log(f"⚠️ 冷却已超 {SLOW_WARN_S:.0f}s 仍 {t}°C，散热可能异常，请检查风扇/机箱", log_fn)
        if elapsed >= COOL_CAP_S:
            _log(f"⚠️ 冷却等待超 {COOL_CAP_S:.0f}s（仍 {t}°C），强制放行（有风险）{tag}", log_fn)
            return
        if elapsed - last_note >= 15:
            last_note = elapsed
            _log(f"仍在冷却：{t}°C → 目标 {RESUME_C:.0f}°C（已等 {elapsed:.0f}s）", log_fn)
        time.sleep(SAMPLE_SEC)


# ---- continuous background monitor ---------------------------------------
class GpuMonitor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="gpu-monitor")
        self.last = None
        self.peak = None
        self.state = "unknown"   # normal | hot | critical | nogpu
        self.started_at = time.monotonic()
        self.last_emergency_at = 0.0
        self._stop = threading.Event()

    def status(self):
        return {
            "enabled": ENABLED,
            "temp": self.last,
            "peak": self.peak,
            "state": self.state,
            "trip_c": TRIP_C,
            "resume_c": RESUME_C,
            "hard_ceil_c": HARD_CEIL_C,
            "uptime_s": round(time.monotonic() - self.started_at, 1),
        }

    def stop(self):
        self._stop.set()

    def run(self):
        if not ENABLED:
            self.state = "disabled"
            return
        was_hot = False
        while not self._stop.is_set():
            t = gpu_temp()
            self.last = t
            if t is None:
                self.state = "nogpu"
            else:
                self.peak = t if self.peak is None else max(self.peak, t)
                if t >= HARD_CEIL_C:
                    self.state = "critical"
                    # rate-limit the loud action to once per 10s
                    if time.monotonic() - self.last_emergency_at > 10:
                        self.last_emergency_at = time.monotonic()
                        _log(f"🔥 紧急：GPU {t}°C ≥ 硬顶 {HARD_CEIL_C:.0f}°C，掐断进行中的 GPU 任务以保护显卡！")
                        _fire_aborts()
                elif t >= TRIP_C:
                    self.state = "hot"
                    if not was_hot:
                        was_hot = True
                        _log(f"GPU 升到 {t}°C（≥{TRIP_C:.0f}°C 警戒）")
                else:
                    self.state = "normal"
                    if was_hot and t <= RESUME_C:
                        was_hot = False
                        _log(f"GPU 已回到 {t}°C（≤{RESUME_C:.0f}°C 安全）")
            self._stop.wait(SAMPLE_SEC)


_monitor = None
_monitor_lock = threading.Lock()


def start_monitor():
    """Start (once) the background temperature monitor; returns the singleton."""
    global _monitor
    with _monitor_lock:
        if _monitor is None and ENABLED:
            _monitor = GpuMonitor()
            _monitor.start()
            _log(f"温控已启动：警戒 {TRIP_C:.0f}°C / 恢复 {RESUME_C:.0f}°C / 硬顶 {HARD_CEIL_C:.0f}°C")
        return _monitor


def get_status():
    if _monitor is not None:
        return _monitor.status()
    return {"enabled": ENABLED, "temp": gpu_temp(), "state": "not-started",
            "trip_c": TRIP_C, "resume_c": RESUME_C, "hard_ceil_c": HARD_CEIL_C}
