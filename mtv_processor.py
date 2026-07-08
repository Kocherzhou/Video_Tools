"""MTV Generation Processor — beat-synced visual montage + music."""

import os
import subprocess
import threading
from typing import Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

ALLOWED_VIDEO_EXTS  = {'.mp4', '.mov', '.avi', '.mkv'}
ALLOWED_IMAGE_EXTS  = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif', '.tiff'}
ALLOWED_PDF_EXTS    = {'.pdf'}
ALLOWED_MUSIC_EXTS  = {'.mp3', '.wav', '.aac', '.flac', '.m4a', '.ogg'}


# ─── Capability checks ────────────────────────────────────────────────────────

def librosa_available() -> bool:
    try:
        import librosa  # noqa: F401
        return True
    except ImportError:
        return False


def pymupdf_available() -> bool:
    try:
        import fitz  # noqa: F401
        return True
    except ImportError:
        return False


def pdf2image_available() -> bool:
    """pdf2image (pure Python) + poppler binaries on PATH."""
    try:
        import pdf2image  # noqa: F401
        # Also verify pdftoppm is reachable
        r = subprocess.run(['pdftoppm', '-v'], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def pdf_support_available() -> bool:
    """True if any PDF backend is available."""
    return pymupdf_available() or pdf2image_available()


# ─── Beat detection ───────────────────────────────────────────────────────────

def _get_duration(path: str) -> float:
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', path],
        capture_output=True, text=True, timeout=15,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def detect_beats(audio_path: str) -> Tuple[List[float], float, float]:
    """Returns (beat_times, duration_sec, bpm).

    Priority:
      1. librosa (most accurate) — used if importable
      2. numpy energy-envelope method (good for rhythmic music)
      3. Fixed 120 BPM fallback
    """
    duration = _get_duration(audio_path)

    # ── 1. Try librosa ────────────────────────────────────────────
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        bpm = float(np.atleast_1d(tempo)[0])
        if duration <= 0:
            duration = float(len(y)) / sr if sr else 0.0   # ffprobe failed; derive from samples
        return beat_times, duration, bpm
    except Exception:
        # NOT just ImportError: a librosa/numba/codec runtime error (corrupt or
        # unsupported audio, version mismatch) must also fall through to the
        # numpy + 120BPM fallbacks below instead of killing the whole MTV job.
        pass

    # ── 2. numpy energy-envelope beat detection ───────────────────
    try:
        beats, bpm = _detect_beats_numpy(audio_path, duration)
        if beats:
            return beats, duration, bpm
    except Exception:
        pass

    # ── 3. Fixed 120 BPM fallback ────────────────────────────────
    bpm = 120.0
    interval = 60.0 / bpm
    n = max(1, int(duration / interval))
    return [i * interval for i in range(1, n + 1)], duration, bpm


def _detect_beats_numpy(audio_path: str, duration: float) -> Tuple[List[float], float]:
    """Energy-envelope beat detection using stdlib + numpy only.

    Steps:
      1. Convert audio to mono 22050 Hz WAV via ffmpeg
      2. Load raw PCM samples with stdlib wave
      3. Compute short-time energy, smooth, find onset peaks
      4. Cluster peaks to estimate BPM, return beat grid
    """
    import numpy as np
    import wave as wave_module
    import tempfile, os

    SR = 22050
    WAV_TMP = tempfile.mktemp(suffix='.wav')
    try:
        r = subprocess.run([
            'ffmpeg', '-y', '-i', audio_path,
            '-ac', '1', '-ar', str(SR),
            '-sample_fmt', 's16', WAV_TMP,
        ], capture_output=True, timeout=60)
        if r.returncode != 0:
            return [], 120.0

        with wave_module.open(WAV_TMP, 'r') as wf:
            raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        try:
            os.remove(WAV_TMP)
        except OSError:
            pass

    if len(samples) < SR:
        return [], 120.0

    # Short-time energy (25 ms window, 10 ms hop)
    win  = int(SR * 0.025)
    hop  = int(SR * 0.010)
    n_fr = (len(samples) - win) // hop
    energy = np.array([
        float(np.sum(samples[i * hop: i * hop + win] ** 2))
        for i in range(n_fr)
    ], dtype=np.float32)

    # Smooth energy with 100 ms window
    sm_w = max(1, int(0.1 * SR / hop))
    kernel = np.ones(sm_w, dtype=np.float32) / sm_w
    smooth = np.convolve(energy, kernel, mode='same')

    # Find onset peaks: local maxima above 1.3× smoothed background
    min_gap = max(1, int(0.18 * SR / hop))  # ≥180 ms between beats
    onsets: List[int] = []
    for i in range(1, n_fr - 1):
        if (energy[i] > energy[i - 1]
                and energy[i] > energy[i + 1]
                and energy[i] > smooth[i] * 1.25):
            if not onsets or i - onsets[-1] >= min_gap:
                onsets.append(i)

    if len(onsets) < 2:
        return [], 120.0

    beat_times = [float(o * hop / SR) for o in onsets]

    # Estimate BPM from median inter-onset interval
    intervals = np.diff(beat_times)
    med = float(np.median(intervals))
    ok  = intervals[(intervals > med * 0.5) & (intervals < med * 2.0)]
    bpm = float(60.0 / np.mean(ok)) if len(ok) else 120.0

    return beat_times, bpm


# ─── PDF conversion ───────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: str, out_dir: str) -> List[str]:
    """Convert PDF pages to PNG images.
    Tries pymupdf first, then pdf2image (requires poppler on PATH)."""

    # ── Backend 1: pymupdf ────────────────────────────────────────
    try:
        import fitz
        doc = fitz.open(pdf_path)
        paths = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            p = os.path.join(out_dir, f'pdf_{i:04d}.png')
            pix.save(p)
            paths.append(p)
        doc.close()
        return paths
    except ImportError:
        pass

    # ── Backend 2: pdf2image (uses poppler's pdftoppm) ────────────
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=150)
        paths = []
        for i, img in enumerate(images):
            p = os.path.join(out_dir, f'pdf_{i:04d}.png')
            img.save(p, 'PNG')
            paths.append(p)
        return paths
    except ImportError:
        pass

    raise RuntimeError(
        'PDF 转换需要 pymupdf 或 pdf2image+poppler。\n'
        '请运行: pip install pdf2image  (需要系统已安装 poppler)'
    )


# ─── Timing plan ──────────────────────────────────────────────────────────────

def build_switch_segments(
    beat_times: List[float],
    n_items: int,
    loops: int,
    music_duration: float,
) -> List[Tuple[float, float]]:
    """Divide music_duration into n_items×loops equal slots.

    Each visual item gets one slot per loop (so each item appears `loops` times).
    Ideal cut boundaries are snapped to the nearest beat for musical feel,
    but never drift more than 40% of the slot duration from the ideal point.
    """
    n_segs = max(1, n_items * loops)
    ideal_interval = music_duration / n_segs
    ideal_pts = [i * ideal_interval for i in range(n_segs + 1)]

    if beat_times:
        snap_limit = ideal_interval * 0.4
        snapped = [0.0]
        for pt in ideal_pts[1:-1]:
            nearest = min(beat_times, key=lambda t: abs(t - pt))
            snapped.append(nearest if abs(nearest - pt) <= snap_limit else pt)
        snapped.append(music_duration)
    else:
        snapped = ideal_pts

    return [
        (snapped[i], snapped[i + 1])
        for i in range(len(snapped) - 1)
        if snapped[i + 1] - snapped[i] > 0.05
    ]


# ─── ffmpeg segment helpers ───────────────────────────────────────────────────

def _vf(w: int, h: int, fps: int) -> str:
    # Blur-fill: same image blurred+scaled to cover background, clear image centered on top.
    # Eliminates black bars for portrait sources; landscape sources are unaffected.
    return (
        f"split=2[fg][bg];"
        f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease[fgs];"
        f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},gblur=sigma=25[bgs];"
        f"[bgs][fgs]overlay=(W-w)/2:(H-h)/2,"
        f"fps={fps},format=yuv420p"
    )


def _make_img_seg(src: str, dur: float, out: str, w: int, h: int, fps: int) -> None:
    # 1 fps input → fps filter resamples to target fps; duplicate P-frames encode in
    # near-zero time, giving ~20x speedup on slow/emulated hardware (ARM64 Windows).
    # Write to a temp path and os.replace() on success so a timed-out/failed ffmpeg
    # never leaves a truncated, non-finalized .mp4 (no moov atom) at `out` that could
    # poison the concat step.
    #
    # `-t` MUST come AFTER `-i` (output option), with `-loop 1` looping the image
    # forever. As an INPUT option it limits how much of the 1 fps source is read:
    # `dur` seconds at 1 fps = `dur` frames spanning [0, dur-1], so the fps filter
    # emits only ~(dur-1)s and every page lands ~1s short. Those deficits accumulate,
    # so the concatenated video ends short of the song; the final `-t music_duration`
    # mux then freezes the LAST page on screen to fill the gap — the "last page is
    # way too long" bug. As an output option `-t` cuts at exactly `dur`.
    timeout = max(60, int(dur * 10))
    tmp = out + '.tmp.mp4'
    try:
        subprocess.run([
            'ffmpeg', '-y', '-loop', '1', '-framerate', '1',
            '-i', src,
            '-t', f'{dur:.4f}',
            '-vf', _vf(w, h, fps),
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '24', '-an',
            tmp,
        ], capture_output=True, timeout=timeout, check=True)
        os.replace(tmp, out)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _make_vid_seg(src: str, ss: float, dur: float, out: str, w: int, h: int, fps: int,
                  loop: bool = False) -> None:
    # loop=True: source video shorter than required dur → -stream_loop -1 fills the gap.
    # Write to a temp path and os.replace() on success so a timed-out/failed ffmpeg
    # never leaves a truncated, non-finalized .mp4 at `out` that could poison concat.
    tmp = out + '.tmp.mp4'
    cmd = ['ffmpeg', '-y']
    if loop:
        cmd += ['-stream_loop', '-1']
    cmd += [
        '-ss', f'{ss:.4f}',
        '-i', src,
        '-t', f'{dur:.4f}',
        '-vf', _vf(w, h, fps),
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '24', '-an',
        tmp,
    ]
    timeout = max(120, int(dur * 15))
    try:
        subprocess.run(cmd, capture_output=True, timeout=timeout, check=True)
        os.replace(tmp, out)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ─── Main pipeline ────────────────────────────────────────────────────────────

def process_mtv_job(
    job_id: str,
    source_infos: List[Dict],
    music_path: str,
    beats_per_switch: int,
    resolution: str,
    work_dir: str,
    update_fn: Callable,  # update_fn(progress, status, message, output_path=None)
    orientation: str = 'landscape',
) -> None:
    """
    Beat-synchronized visual montage pipeline:
      1. Detect beats in audio
      2. Collect & convert visual sources (PDF→PNG, keep images/videos)
      3. Assign each beat-slot to a visual item (cycling)
      4. Render each slot as a short MP4 segment (parallel)
      5. Concat all segments
      6. Mux with music track
    """
    W, H = (1920, 1080) if resolution == '1080p' else (1280, 720)
    if orientation == 'portrait':
        W, H = H, W   # swap to vertical (e.g. 1080x1920) — fills phone screenshots
    FPS = 25

    try:
        # ── 1. Detect beats ───────────────────────────────────────────
        update_fn(5, 'analyzing', '正在分析音乐节拍...')
        beat_times, music_duration, tempo = detect_beats(music_path)
        if music_duration <= 0:
            # Otherwise build_switch_segments collapses to [] and the job dies
            # later with an opaque "片段合并失败"; fail clearly here instead.
            raise RuntimeError('无法读取音乐时长（音频文件可能损坏或格式不支持）')
        update_fn(15, 'analyzing',
                  f'检测到 {len(beat_times)} 个节拍，BPM ≈ {tempo:.0f}，时长 {music_duration:.1f}s')

        # ── 2. Prepare visual items ───────────────────────────────────
        update_fn(18, 'preparing', '正在准备视觉素材...')
        visual_items: List[Dict] = []
        pdf_dir = os.path.join(work_dir, 'pdf_pages')
        os.makedirs(pdf_dir, exist_ok=True)

        for src in source_infos:
            if src['type'] == 'pdf':
                pages = pdf_to_images(src['path'], pdf_dir)
                visual_items.extend({'type': 'image', 'path': p} for p in pages)
            elif src['type'] == 'image':
                visual_items.append({'type': 'image', 'path': src['path']})
            elif src['type'] == 'video':
                dur = _get_duration(src['path'])
                visual_items.append({'type': 'video', 'path': src['path'], 'duration': dur})

        if not visual_items:
            raise ValueError('没有有效的视觉素材')

        n_items = len(visual_items)
        loops   = max(1, beats_per_switch)  # repurposed: how many times to cycle through items
        avg_dur = music_duration / (n_items * loops)
        update_fn(25, 'preparing',
                  f'已准备 {n_items} 个素材，每项平均时长 {avg_dur:.1f}s，循环 {loops} 次')

        # ── 3. Build switch-point timing ──────────────────────────────
        # n_items × loops equal slots, cut points snapped to nearest beat
        segs   = build_switch_segments(beat_times, n_items, loops, music_duration)
        n_segs = len(segs)
        update_fn(28, 'rendering',
                  f'共 {n_segs} 个片段（{n_items} 项 × {loops} 次循环），节拍对齐切点')

        # ── 4. Assign items and build render tasks ────────────────────
        segs_dir = os.path.join(work_dir, 'segs')
        os.makedirs(segs_dir, exist_ok=True)

        vid_pos: Dict[str, float] = {
            it['path']: 0.0 for it in visual_items if it['type'] == 'video'
        }

        tasks: List[Tuple] = []  # (index, item, ss, dur, out_path)
        for i, (t0, t1) in enumerate(segs):
            dur  = t1 - t0
            item = visual_items[i % n_items]   # cycle through items per loop
            out  = os.path.join(segs_dir, f's{i:06d}.mp4')
            if item['type'] == 'video':
                vp = vid_pos[item['path']]
                vd = item.get('duration', 10.0)
                if vp >= vd:
                    vp = 0.0
                    vid_pos[item['path']] = 0.0
                tasks.append((i, item, vp, dur, out))
                vid_pos[item['path']] = vp + dur
            else:
                tasks.append((i, item, 0.0, dur, out))

        # ── 4b. Render segments in parallel ──────────────────────────
        done = [0]
        lock = threading.Lock()

        def _render(task: Tuple) -> Tuple[int, str]:
            idx, item, ss, dur, out_path = task
            if item['type'] == 'image':
                _make_img_seg(item['path'], dur, out_path, W, H, FPS)
            else:
                src_dur = item.get('duration', 0.0)
                need_loop = src_dur > 0 and ss + dur > src_dur
                _make_vid_seg(item['path'], ss, dur, out_path, W, H, FPS, loop=need_loop)
            with lock:
                done[0] += 1
            return idx, out_path

        seg_paths: List[str] = [''] * n_segs
        workers = min(4, os.cpu_count() or 2)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            fmap = {pool.submit(_render, t): t[0] for t in tasks}
            for fut in as_completed(fmap):
                idx, out_path = fut.result()
                seg_paths[idx] = out_path
                pct = 30 + int(done[0] / n_segs * 50)
                update_fn(pct, 'rendering', f'渲染片段 {done[0]}/{n_segs}...')

        # ── 5. Concat segments ────────────────────────────────────────
        update_fn(82, 'merging', '正在拼接片段...')
        concat_txt = os.path.join(segs_dir, 'concat.txt')
        with open(concat_txt, 'w', encoding='utf-8') as f:
            for p in seg_paths:
                f.write(f"file '{os.path.basename(p)}'\n")

        raw_mp4 = os.path.abspath(os.path.join(work_dir, 'raw.mp4')).replace('\\', '/')
        r = subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', 'concat.txt', '-c', 'copy', raw_mp4,
        ], capture_output=True, timeout=600, cwd=segs_dir)
        if r.returncode != 0:
            raise RuntimeError('片段合并失败：' + r.stderr.decode('utf-8', errors='replace')[-300:])

        # ── 6. Add music ──────────────────────────────────────────────
        update_fn(92, 'finalizing', '正在添加音乐轨...')
        out_mp4 = os.path.join(work_dir, 'output.mp4')
        r = subprocess.run([
            'ffmpeg', '-y',
            '-i', raw_mp4,
            '-i', os.path.abspath(music_path),
            '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
            '-t', f'{music_duration:.4f}',
            os.path.abspath(out_mp4),
        ], capture_output=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError('添加音乐失败：' + r.stderr.decode('utf-8', errors='replace')[-300:])

        update_fn(100, 'completed',
                  f'MTV 生成完成！时长 {music_duration:.1f}s，共 {n_segs} 个切换',
                  os.path.abspath(out_mp4))

    except Exception as exc:
        update_fn(0, 'error', f'生成失败：{exc}')
