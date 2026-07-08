#!/usr/bin/env python3
"""gpu_watch.py — standalone live GPU thermal monitor for the RTX 3080.

A tiny terminal program you can run in its own window to keep an eye on the
card while subtitle/translation jobs run. Reuses gpu_guard's thresholds so the
colours match exactly what the in-app shield reacts to:

  green  < RESUME_C (55)         safe
  yellow RESUME_C..TRIP_C        warming
  red    >= TRIP_C (65)          shield would pause new jobs here
  bold   >= HARD_CEIL_C (88)     emergency — shield aborts in-flight jobs

Usage:  python3 gpu_watch.py [interval_seconds]   (default 2s)

Reads temperature directly via nvidia-smi (no server needed). Ctrl-C to quit.
"""
import sys
import time

import gpu_guard

INTERVAL = 2.0
if len(sys.argv) > 1:
    try:
        INTERVAL = float(sys.argv[1])
    except ValueError:
        pass

G, Y, R, B, X = "\033[32m", "\033[33m", "\033[31m", "\033[1;31m", "\033[0m"


def colour(t):
    if t is None:
        return X, "无信号"
    if t >= gpu_guard.HARD_CEIL_C:
        return B, "熔断"
    if t >= gpu_guard.TRIP_C:
        return R, "过热·暂停区"
    if t >= gpu_guard.RESUME_C:
        return Y, "升温"
    return G, "安全"


def bar(t):
    if t is None:
        return " " * 30
    lo, hi = 35, 95
    n = max(0, min(30, round((t - lo) / (hi - lo) * 30)))
    return "█" * n + "·" * (30 - n)


def main():
    print(f"GPU 温控监视 — 警戒 {gpu_guard.TRIP_C:.0f}°C / 恢复 {gpu_guard.RESUME_C:.0f}°C / "
          f"硬顶 {gpu_guard.HARD_CEIL_C:.0f}°C，每 {INTERVAL:.0f}s 刷新，Ctrl-C 退出\n")
    peak = None
    try:
        while True:
            t = gpu_guard.gpu_temp()
            if t is not None:
                peak = t if peak is None else max(peak, t)
            c, label = colour(t)
            shown = f"{t:>3}°C" if t is not None else " N/A "
            pk = f"{peak:>3}°C" if peak is not None else " N/A "
            ts = time.strftime("%H:%M:%S")
            sys.stdout.write(f"\r{ts}  {c}{shown}{X} [{c}{bar(t)}{X}] {c}{label:<10}{X} 峰值 {pk}   ")
            sys.stdout.flush()
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    main()
