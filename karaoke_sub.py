#!/usr/bin/env python3
r"""karaoke_sub.py — 给成片 mp4 烧「逐字卡拉OK」字幕(用已知歌词原文,非听写)。

对齐:faster-whisper 取「唱歌时段」(词级时间;设备可配 cpu/cuda)→ 把已知歌词行
按字数比例铺到唱歌时段上(跳过前奏/间奏静默)→ 每行内逐字 \kf 高亮 → ffmpeg 烧录。

设备:--device cpu(默认,3080 没修时用)/ cuda(3080 修好后,快)。
样式:--font-size --margin-v(离底) --outline(描边) --primary(已唱色) --secondary(未唱色)。

用法:
  python karaoke_sub.py --video 成片.mp4 --lyrics 歌词.txt --out 带字幕.mp4 [--device cpu]
"""
import argparse, os, subprocess, sys
from pathlib import Path

FF = os.environ.get("FFDIR", "/home/user/miniconda3/bin")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def probe(path, args):
    return run([f"{FF}/ffprobe", "-v", "error", *args, str(path)]).stdout.strip()


def dims(v):
    w, h = probe(v, ["-select_streams", "v:0", "-show_entries", "stream=width,height",
                     "-of", "csv=p=0"]).split(",")[:2]
    return int(w), int(h)


def sung_spans(wav, device, model_size):
    """返回唱歌时段列表 [(start,end), ...](合并相邻、过滤极短),来自 whisper 段时间。"""
    from faster_whisper import WhisperModel
    ct = "int8" if device == "cpu" else "float16"
    m = WhisperModel(model_size, device=device, compute_type=ct)
    # ⚠️ 不用 vad_filter:VAD 为说话调,会把背景乐里的唱歌当噪音滤掉(实测 225s 歌只剩 22s)。
    # condition_on_previous_text=False:减少在纯器乐段的重复幻觉。
    segs, _ = m.transcribe(wav, language="zh", word_timestamps=True,
                           vad_filter=False, condition_on_previous_text=False)
    spans = []
    for s in segs:
        # 过滤器乐段的幻觉:no_speech_prob 高 / 置信度过低 / 极短,都不是真唱
        nsp = getattr(s, "no_speech_prob", 0.0)
        alp = getattr(s, "avg_logprob", 0.0)
        if nsp > 0.6 or alp < -1.0 or (float(s.end) - float(s.start)) < 0.3:
            continue
        spans.append((float(s.start), float(s.end)))
    if not spans:                                  # 全被过滤则退回原始(保底不崩)
        spans = [(float(s.start), float(s.end)) for s in []] or [(0.0, 1.0)]
    # 合并间隔 <0.4s 的相邻段
    merged = []
    for st, en in spans:
        if merged and st - merged[-1][1] < 0.4:
            merged[-1] = (merged[-1][0], en)
        else:
            merged.append((st, en))
    return [(a, b) for a, b in merged if b - a > 0.2]


def map_to_timeline(spans, total_chars_cum, line_chars):
    """把歌词行(按累计字数)铺到唱歌时段拼成的连续时间线上,返回每行 (start,end)。"""
    sung = sum(b - a for a, b in spans)
    total = total_chars_cum[-1]

    def frac_to_time(frac):
        target = frac * sung
        acc = 0.0
        for a, b in spans:
            seg = b - a
            if acc + seg >= target:
                return a + (target - acc)
            acc += seg
        return spans[-1][1]

    out = []
    c0 = 0
    for n in line_chars:
        s = frac_to_time(c0 / total)
        e = frac_to_time((c0 + n) / total)
        out.append((s, max(e, s + 0.4)))
        c0 += n
    return out


def cs(t):  # 秒 → ASS 时间 h:mm:ss.cc
    t = max(0.0, t); h = int(t // 3600); m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_ass(lines, times, W, H, font, fsize, marginv, outline, primary, secondary):
    # ASS 颜色 &HAABBGGRR;传入 #RRGGBB → 转
    def col(hexrgb):
        hexrgb = hexrgb.lstrip("#")
        r, g, b = hexrgb[0:2], hexrgb[2:4], hexrgb[4:6]
        return f"&H00{b}{g}{r}".upper()
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: K,{font},{fsize},{col(primary)},{col(secondary)},&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,{outline},1,2,40,40,{marginv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, Effect, Text
"""
    ev = []
    for (txt, (st, en)) in zip(lines, times):
        chars = [c for c in txt]
        per = max(1, int(round((en - st) * 100 / len(chars))))  # 每字 centiseconds
        kara = "".join(f"{{\\kf{per}}}{c}" for c in chars)
        ev.append(f"Dialogue: 0,{cs(st)},{cs(en)},K,,0,0,0,,{kara}")
    return head + "\n".join(ev) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--lyrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=os.environ.get("KARAOKE_DEVICE", "cpu"))
    ap.add_argument("--model", default=os.environ.get("WHISPER_MODEL_SIZE", "large-v3-turbo"))
    ap.add_argument("--font", default="Noto Sans CJK SC")
    ap.add_argument("--font-size", type=int, default=54)
    ap.add_argument("--margin-v", type=int, default=70)
    ap.add_argument("--outline", type=float, default=3.0)
    ap.add_argument("--primary", default="#33E0FF")     # 已唱(扫过)色
    ap.add_argument("--secondary", default="#FFFFFF")   # 未唱色
    args = ap.parse_args()

    lines = [l.strip() for l in Path(args.lyrics).read_text(encoding="utf-8").splitlines() if l.strip()]
    line_chars = [len(l) for l in lines]
    cum = [0]
    for n in line_chars:
        cum.append(cum[-1] + n)

    W, H = dims(args.video)
    wav = "/tmp/_kara.wav"
    print(f"[1/4] 抽音轨…");
    run([f"{FF}/ffmpeg", "-y", "-i", args.video, "-vn", "-ac", "1", "-ar", "16000", wav])
    print(f"[2/4] Whisper 取唱歌时段(device={args.device})…")
    spans = sung_spans(wav, args.device, args.model)
    print(f"      检测到 {len(spans)} 个唱歌时段,总唱 {sum(b-a for a,b in spans):.1f}s")
    times = map_to_timeline(spans, cum, line_chars)
    print(f"[3/4] 生成卡拉OK ASS({len(lines)} 行)…")
    ass = build_ass(lines, times, W, H, args.font, args.font_size, args.margin_v,
                    args.outline, args.primary, args.secondary)
    ass_path = "/tmp/_kara.ass"
    Path(ass_path).write_text(ass, encoding="utf-8")
    print(f"[4/4] 烧录…")
    r = run([f"{FF}/ffmpeg", "-y", "-i", args.video, "-vf", f"ass={ass_path}",
             "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
             "-c:a", "copy", args.out])
    if r.returncode != 0 or not os.path.exists(args.out):
        print("烧录失败:", r.stderr[-500:]); sys.exit(1)
    print(f"完成 -> {args.out}  ({os.path.getsize(args.out)//1000000} MB)")


if __name__ == "__main__":
    main()
