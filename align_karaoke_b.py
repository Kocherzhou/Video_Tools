#!/usr/bin/env python3
r"""align_karaoke_b.py — B 方案:中文 wav2vec2-CTC + torchaudio.forced_align,
把【已知歌词原文】逐字强制对齐到音频,生成逐字卡拉OK字幕并烧录。

跑在 infinitetalk 环境(torch2.4.1+cu121 / torchaudio / transformers)。
  /home/user/miniconda3/envs/infinitetalk/bin/python align_karaoke_b.py \
     --video 成片.mp4 --lyrics 歌词.txt --out 带字幕.mp4 [--device cuda]
"""
import argparse, os, re, subprocess, sys
from pathlib import Path
import torch, torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

FF = "/home/user/miniconda3/bin"


def clean_lyrics(path):
    """只留可唱的歌词行:过滤 [段落标记] 与整行(舞台说明);剥掉行首 (男)/(女)/(合) 演唱者标签。"""
    out = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):              # [Intro] / [Verse 1 - 男]
            continue
        s = re.sub(r"^[(（](男|女|合)[)）]\s*", "", s).strip()   # 剥掉演唱者标签,保留歌词
        if not s:
            continue
        if re.match(r"^[(（].*[)）]$", s):                       # 整行括号=舞台说明,跳过
            continue
        out.append(s)
    return out


def dims(v):
    out = subprocess.run([f"{FF}/ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", v],
        capture_output=True, text=True).stdout.strip()
    w, h = out.split(",")[:2]; return int(w), int(h)


def cs(t):
    t = max(0.0, t); h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def col(hexrgb):
    hexrgb = hexrgb.lstrip("#"); r, g, b = hexrgb[0:2], hexrgb[2:4], hexrgb[4:6]
    return f"&H00{b}{g}{r}".upper()


def load_audio(path, sr=16000):
    wav = "/tmp/_b.wav"
    subprocess.run([f"{FF}/ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), wav],
                   capture_output=True)
    w, got = torchaudio.load(wav)
    if got != sr:
        w = torchaudio.functional.resample(w, got, sr)
    return w[0], sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--lyrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
    ap.add_argument("--font", default="Noto Sans CJK SC")
    ap.add_argument("--font-size", type=int, default=54)
    ap.add_argument("--margin-v", type=int, default=70)
    ap.add_argument("--outline", type=float, default=3.0)
    ap.add_argument("--primary", default="#33E0FF")
    ap.add_argument("--secondary", default="#FFFFFF")
    ap.add_argument("--lead-in", type=float, default=0.7)   # 提前出现(秒)
    ap.add_argument("--linger", type=float, default=1.5)    # 延后停留(秒)
    ap.add_argument("--lang", default="zh")                 # zh / en(英文待完善)
    args = ap.parse_args()
    if args.lang == "en" and "chinese" in args.model:
        args.model = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

    dev = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[1/5] 载模型 {args.model} (device={dev})…")
    proc = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(dev).eval()
    vocab = proc.tokenizer.get_vocab()           # token -> id
    blank = model.config.pad_token_id or 0

    print("[2/5] 抽音轨 + 分块出帧概率(20s/块,防 OOM + 短 GPU 爆发)…")
    wav, sr = load_audio(args.video)
    chunk = 20 * sr                               # 20 秒一块
    embs = []
    with torch.inference_mode():
        for i in range(0, wav.size(0), chunk):
            seg = wav[i:i + chunk].unsqueeze(0).to(dev)
            lg = model(seg).logits                # [1,t,V]
            embs.append(torch.log_softmax(lg, dim=-1)[0].cpu())
            if dev == "cuda":
                torch.cuda.empty_cache()
    emission = torch.cat(embs, dim=0)
    T = emission.size(0); dur = wav.size(0) / sr; ratio = dur / T
    print(f"      帧数 T={T} 时长 {dur:.1f}s 每帧 {ratio*1000:.1f}ms")

    lines = clean_lyrics(args.lyrics)
    # 字符级覆盖率
    allchars = [c for ln in lines for c in ln]
    cover = sum(1 for c in allchars if c in vocab) / max(1, len(allchars))
    print(f"[3/5] 词表覆盖率(字符级)= {cover:.0%} (vocab {len(vocab)})")

    # 构造 target tokens + (line_idx, pos_in_line) 映射;OOV 字跳过、后面插值
    targets, tmap = [], []
    for li, ln in enumerate(lines):
        for pi, ch in enumerate(ln):
            tid = vocab.get(ch)
            if tid is not None and tid != blank:
                targets.append(tid); tmap.append((li, pi))
    if len(targets) < 3:
        print("覆盖率过低,字符级对齐不可用(可能是拼音词表模型)。换字符级中文模型再试。")
        sys.exit(2)

    print(f"[4/5] forced_align {len(targets)} 字…")
    tt = torch.tensor(targets, dtype=torch.int32).unsqueeze(0)
    aligned, scores = torchaudio.functional.forced_align(
        emission.unsqueeze(0), tt, blank=blank)
    spans = torchaudio.functional.merge_tokens(aligned[0], scores[0])
    spans = [s for s in spans if s.token != blank]
    # 对齐后 spans 顺序 == targets 顺序
    char_t = {}                                   # (li,pi) -> (start_s, end_s)
    for (li, pi), sp in zip(tmap, spans):
        char_t[(li, pi)] = (sp.start * ratio, sp.end * ratio)

    # 每行:用已知字时间,OOV 字在行内线性插值;行起止=首尾字
    def line_times(li, ln):
        known = [(pi, char_t[(li, pi)]) for pi in range(len(ln)) if (li, pi) in char_t]
        if not known:
            return None
        res = [None] * len(ln)
        for pi, (s, e) in known:
            res[pi] = [s, e]
        # 头尾补齐
        first_pi = known[0][0]; last_pi = known[-1][0]
        for pi in range(0, first_pi):
            res[pi] = [known[0][1][0], known[0][1][0]]
        for pi in range(last_pi + 1, len(ln)):
            res[pi] = [known[-1][1][1], known[-1][1][1]]
        # 中间空洞线性插值
        pi = 0
        while pi < len(ln):
            if res[pi] is None:
                j = pi
                while j < len(ln) and res[j] is None:
                    j += 1
                a = res[pi - 1][1]; b = res[j][0]; n = j - pi
                for k in range(n):
                    t0 = a + (b - a) * k / n; t1 = a + (b - a) * (k + 1) / n
                    res[pi + k] = [t0, t1]
                pi = j
            else:
                pi += 1
        return res

    print("[5/5] 生成卡拉OK ASS + 烧录…")
    W, H = dims(args.video)
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: K,{args.font},{args.font_size},{col(args.primary)},{col(args.secondary)},&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,{args.outline},1,2,40,40,{args.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, Effect, Text
"""
    LEADIN, LINGER = args.lead_in, args.linger     # 提前出现 / 延后消失(秒)
    rows = []
    for li, ln in enumerate(lines):
        ts = line_times(li, ln)
        if ts:
            rows.append((ln, ts, ts[0][0], ts[-1][1]))   # (文本, 逐字时间, 唱起, 唱止)
    # 显示窗:提前 LEADIN 出现、延后 LINGER 消失,且不与前后行重叠
    disp_s = [max(ss - LEADIN, rows[i-1][3] if i > 0 else 0.0)
              for i, (_, _, ss, _) in enumerate(rows)]
    disp_e = []
    for i, (_, _, ss, se) in enumerate(rows):
        nxt = disp_s[i + 1] if i < len(rows) - 1 else se + LINGER + 1
        disp_e.append(min(se + LINGER, nxt - 0.05))
    ev = []
    for i, (ln, ts, ss, se) in enumerate(rows):
        ds, de = disp_s[i], disp_e[i]
        if de <= ds:
            de = ds + 0.5
        starts = [t[0] for t in ts]
        kara = ""
        lead = int(round((ss - ds) * 100))         # 出现→开唱的空档(此段不高亮)
        if lead > 0:
            kara += f"{{\\k{lead}}}"
        for j, ch in enumerate(ln):
            nxt = starts[j + 1] if j < len(starts) - 1 else se
            d = max(1, int(round((nxt - starts[j]) * 100)))   # 扫光时长=到下个字开唱
            kara += f"{{\\kf{d}}}{ch}"
        ev.append(f"Dialogue: 0,{cs(ds)},{cs(de)},K,,0,0,0,{kara}")
    ass_path = "/tmp/_b.ass"
    Path(ass_path).write_text(head + "\n".join(ev) + "\n", encoding="utf-8")
    # 打印前 3 行时间核验
    for d in ev[:3]:
        print("   ", d.split(",,")[0])

    r = subprocess.run([f"{FF}/ffmpeg", "-y", "-i", args.video, "-vf", f"ass={ass_path}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "copy", args.out], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(args.out):
        print("烧录失败:", r.stderr[-500:]); sys.exit(1)
    print(f"完成 -> {args.out} ({os.path.getsize(args.out)//1000000} MB)")


if __name__ == "__main__":
    main()
