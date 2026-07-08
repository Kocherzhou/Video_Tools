#!/usr/bin/env python3
"""
Gemini 语音识别（识别原文，不翻译）—— 供 app.py 的「中文原文模式」复用。

设计要点（为什么不用 Gemini 出 SRT）：
  Gemini 的中文文本识别很准，但生成 SRT 时间戳极不可靠——长音频会产出畸形时间戳
  （如 00:04:900,000）、整块串行、甚至中途截断，解析后整块丢失、屏上叠字/冻结。
  所以这里**不让 Gemini 做时间戳**：
    1. ffmpeg 把音频切成若干 ~20 秒小块（每块的 [起,止] 区间是已知的）。
    2. 每块只让 Gemini 输出**纯文本转写**（每句一行，不要时间戳/序号），全质量文本。
    3. 在该块的已知时间区间内，按每行字数**等比分配**时间 → 生成 cue。
  好处：永远不会因时间戳畸形而丢句/叠字；连续解说每 20 秒重新对齐一次，同步足够好。
  代价：块内时间是按字数等比的近似（非逐字 VAD 级），但对解说类视频完全够用，
  且用户可在前端直接编辑修正。

只依赖标准库 + ffmpeg/ffprobe。key 由调用方（app.py 的 GEMINI_KEYS 池）传入，
全局轮询：每成功一段就推进到下一个 key，跨块跨 job 持久，请求量在所有 key 间均摊。
"""

import base64
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Tuple

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# 音频转写模型，独立于翻译模型。gemini-2.5-flash 音频能力好、是验证过的默认。
DEFAULT_MODEL = os.environ.get("TRANSCRIBE_MODEL", "gemini-2.5-flash")
# 每段音频时长（秒）。纯文本方案下，块大小不再影响**完整性**（不会丢句），只影响
# 块内等比计时的松紧。60 秒在「请求数（免费配额）」与「计时精度」间取平衡：9 分钟
# 视频仅约 9 个请求，远比 20s 的 ~27 个省配额。可经 TRANSCRIBE_CHUNK_SECONDS 覆盖。
CHUNK_SECONDS = int(os.environ.get("TRANSCRIBE_CHUNK_SECONDS", "60"))
# 全部 key 被限流时的退避重试：最多重试次数与每次等待秒数（骑过免费配额每分钟窗口）。
_RL_MAX_RETRY = int(os.environ.get("TRANSCRIBE_RL_RETRY", "2"))
_RL_WAIT = int(os.environ.get("TRANSCRIBE_RL_WAIT", "25"))

_PROMPT = (
    "Transcribe ALL speech in this audio clip in the ORIGINAL spoken language, as "
    "PLAIN TEXT. If the spoken language is Chinese, you MUST output Simplified "
    "Chinese characters (简体中文), never Traditional. Put each natural sentence or "
    "clause on its own line. Do NOT translate. Do NOT output timestamps, sequence "
    "numbers, speaker labels, brackets, or any SRT/markup. Transcribe the ENTIRE "
    "clip from beginning to end — do not stop early, do not summarize, do not skip "
    "any speech. Ignore pure music or silence. Output ONLY the transcription text."
)


class SubtitleError(Exception):
    """字幕识别过程中的可预期错误。"""


class RateLimited(SubtitleError):
    """本段所有可用 key 都被限流（429/403）——调用方可退避后重试。"""


# 全局轮询游标：进程级，跨块、跨 job 持久。每成功识别一段就推进到下一个 key，
# 让请求量在所有 key 间均摊（避开单 key 的 RPM 限流）。多 job 并发用锁保护。
_key_cursor = 0
_cursor_lock = threading.Lock()
# 失效 key（返回过 HTTP 400 = 无效/格式错）的下标集合，本进程内永久跳过，
# 不再每轮都浪费一次请求。429/403 是暂时限流，不拉黑。
_dead_keys: set = set()
_dead_lock = threading.Lock()


def _cursor_take(n: int) -> int:
    with _cursor_lock:
        return _key_cursor % n


def _cursor_advance(used_index: int, n: int) -> None:
    global _key_cursor
    with _cursor_lock:
        _key_cursor = (used_index + 1) % n


def _is_dead(ki: int) -> bool:
    with _dead_lock:
        return ki in _dead_keys


def _mark_dead(ki: int) -> None:
    with _dead_lock:
        _dead_keys.add(ki)


# 繁→简兜底：即使 prompt 要求简体，Gemini 偶尔仍输出繁体，这里用纯 Python 的
# zhconv（无需编译，ARM64 安全）再转一道。未安装则保持原样（prompt 已尽量保证）。
try:
    import zhconv as _zhconv
except ImportError:
    _zhconv = None


def to_simplified(text: str) -> str:
    if _zhconv and text:
        try:
            return _zhconv.convert(text, "zh-cn")
        except Exception:
            return text
    return text


# 半角→全角标点映射（Gemini 在长音频后半段常混入半角 , . ? ! 等）
_FULL_PUNCT = {",": "，", ".": "。", "?": "？", "!": "！",
               ":": "：", ";": "；", "(": "（", ")": "）"}
# 末尾要去掉的标点：仅断句类（逗号/顿号/分号/冒号，半全角都含）+ 空白。
# 句末终结标点「。．. ！？!?」一律保留 —— 它们是 resegment_by_sentence 断句的依据，
# 剥掉「。」会让以句号结尾的句子丢失边界、配音时与下句错误合并（老 bug 根因）。
_TRAIL_PUNCT = "，,、；;：:　 \t"
_LEAD_PUNCT = "，,、；;　 \t"


def _is_cjk(ch: str) -> bool:
    return bool(ch) and "㐀" <= ch <= "鿿"


def normalize_punct(text: str) -> str:
    """半角标点→全角，但**仅当与中文相邻**时转，避免误伤数字（4.5、1,000）和英文缩写。"""
    if not text:
        return text
    n = len(text)
    out = []
    for i, ch in enumerate(text):
        if ch in _FULL_PUNCT:
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < n else ""
            if _is_cjk(prev) or _is_cjk(nxt):
                out.append(_FULL_PUNCT[ch])
                continue
        out.append(ch)
    return "".join(out)


def strip_edge_punct(text: str) -> str:
    """去掉每条字幕首尾多余的标点（如句末句号、断句逗号），保留 ？！ 等语气标点。"""
    t = (text or "").strip()
    t = t.rstrip(_TRAIL_PUNCT)
    t = t.lstrip(_LEAD_PUNCT)
    return t.strip()


# ------------------------------------------------------------------- ffmpeg io

def _run(cmd: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _ensure_ffmpeg():
    for tool in ("ffmpeg", "ffprobe"):
        try:
            _run([tool, "-version"], timeout=20)
        except (OSError, subprocess.SubprocessError) as e:
            raise SubtitleError(f"未找到 {tool}，无法识别字幕：{e}")


def _ffprobe_duration(path: Path) -> float:
    r = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(path),
    ], timeout=60)
    try:
        return float(r.stdout.strip())
    except (ValueError, AttributeError):
        raise SubtitleError(f"无法读取视频时长：{r.stderr.strip()[:200]}")


def _extract_chunks(video_path: Path, workdir: Path,
                    chunk_seconds: int) -> List[Tuple[float, float, Path]]:
    """抽取音频并切段，返回 [(起始秒, 结束秒, 段文件路径), ...]（16k 单声道 mp3）。"""
    duration = _ffprobe_duration(video_path)
    chunks: List[Tuple[float, float, Path]] = []
    start, idx = 0.0, 0
    while start < duration:
        end = min(start + chunk_seconds, duration)
        out = workdir / f"chunk_{idx:03d}.mp3"
        r = _run([
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{chunk_seconds}",
            "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
            "-b:a", "48k", str(out),
        ], timeout=600)
        if out.exists() and out.stat().st_size > 0:
            chunks.append((start, end, out))
        elif idx == 0:
            raise SubtitleError(f"音频抽取失败：{r.stderr.strip()[:200]}")
        start += chunk_seconds
        idx += 1
    if not chunks:
        raise SubtitleError("未能从视频中抽取到音频")
    return chunks


# ------------------------------------------------------------------ gemini api

def _extract_text(resp: dict) -> str:
    try:
        parts = resp["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, TypeError):
        return ""


def _gemini_transcribe(audio_path: Path, keys: List[str], model: str,
                       start: int = 0, timeout: int = 300,
                       log: Optional[Callable[[str], None]] = None,
                       prompt: str = None,
                       max_tokens: int = 8192) -> Tuple[str, int]:
    """
    把一段音频 + 文本 prompt 发给 Gemini，返回 (响应文本, 成功 key 的下标)。
    从 keys[start] 开始，遇 429/403/400 环形轮换；调用方用返回下标做全局轮询。
    """
    if not keys:
        raise SubtitleError("未配置 Gemini API key（GEMINI_KEYS 池为空）")

    data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    body = {
        "contents": [{
            "parts": [
                {"text": prompt or _PROMPT},
                {"inline_data": {"mime_type": "audio/mp3", "data": data}},
            ]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": max_tokens,
            # 转写不需要思考；2.5-flash 默认动态思考会吃掉输出预算导致截断，显式关闭。
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    payload = json.dumps(body).encode("utf-8")
    last_err = "未知错误"
    n = len(keys)

    # 本次尝试的 key 顺序：从 start 起环形，跳过已拉黑的失效 key。
    order = [ki for ki in ((start + off) % n for off in range(n)) if not _is_dead(ki)]
    if not order:
        raise SubtitleError("所有 key 均已失效（HTTP 400），请检查 gemini_keys.txt")

    saw_ratelimit = False
    for ki in order:
        key = keys[ki]
        url = GEMINI_ENDPOINT.format(model=model) + "?key=" + key
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url, data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    resp = json.loads(r.read().decode("utf-8"))
                text = _extract_text(resp)
                if text.strip():
                    return text, ki
                last_err = "Gemini 返回空内容"
                break  # 换下一个 key
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "ignore")[:200]
                last_err = f"HTTP {e.code}: {detail}"
                if e.code == 400:
                    _mark_dead(ki)   # 无效 key -> 永久拉黑，不再浪费请求
                    if log:
                        log(f"key #{ki + 1} 无效（HTTP 400），已拉黑跳过")
                    break
                if e.code in (429, 403):
                    saw_ratelimit = True
                    if log:
                        log(f"key #{ki + 1} 限流（HTTP {e.code}），切换下一个 key...")
                    break  # 暂时限流 -> 换下一个 key
                time.sleep(2 * (attempt + 1))  # 5xx -> 退避重试同一 key
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = str(e)
                time.sleep(2 * (attempt + 1))
    if saw_ratelimit:
        raise RateLimited(last_err)   # 全部可用 key 都被限流 -> 让调用方退避重试
    raise SubtitleError(f"Gemini 转写失败：{last_err}")


# --------------------------------------------------------------- text -> cues

def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t.strip())
    return t.strip()


# 残留的非语音标记 / 纯标点行，过滤掉（prompt 已要求忽略，这里兜底）
_NONSPEECH_RE = re.compile(r"^[\s\[\(（【♪♫\-—.。·…]*$")
# 整行被括号包裹的，几乎都是 [音乐]/(掌声)/【音乐】 这类非语音标记
_BRACKET_RE = re.compile(r"^[\[\(（【][^\]\)）】]{0,12}[\]\)）】]$")
_LEAD_JUNK_RE = re.compile(r"^\s*(?:\d+[\.\)、]?\s*|[-—*·•]\s*)")  # 行首序号/项目符号


def _clean_lines(text: str) -> List[str]:
    """把 Gemini 的纯文本拆成干净的字幕行。"""
    lines = []
    for raw in text.splitlines():
        ln = re.sub(r"\s+", " ", raw).strip()
        ln = _LEAD_JUNK_RE.sub("", ln)
        # 去掉残留的时间戳/箭头碎片（极少数情况下 Gemini 仍会冒出来）
        ln = re.sub(r"\d{1,2}:\d{1,2}:\d{1,2}[,.]\d{1,3}", "", ln)
        ln = ln.replace("-->", "").strip()
        if not ln or _NONSPEECH_RE.match(ln) or _BRACKET_RE.match(ln):
            continue
        lines.append(ln)
    return lines


def _chunk_to_cues(text: str, start: float, end: float) -> List[List]:
    """把一段纯文本按行字数在 [start, end] 内等比分配时间，返回 [[s, e, body], ...]。"""
    lines = _clean_lines(text)
    if not lines:
        return []
    total = sum(len(l) for l in lines) or 1
    span = max(0.0, end - start)
    cues: List[List] = []
    t = start
    for i, ln in enumerate(lines):
        seg = span * (len(ln) / total)
        e = end if i == len(lines) - 1 else t + seg
        if e <= t:
            e = t + 0.4
        cues.append([t, e, ln])
        t = e
    return cues


# ---------------------------------------------------------------- public api

def transcribe_to_subtitles(video_path, keys: List[str], model: str = DEFAULT_MODEL,
                            chunk_seconds: int = CHUNK_SECONDS,
                            progress: Optional[Callable[[str], None]] = None) -> List[dict]:
    """
    识别原文（纯文本，块内等比计时）→ 产出 app 的字幕结构
    [{start,end,original,translation}]。识别文本同时写入 original 和 translation：
    build_ass 烧的是 translation，预览/SRT 导出也读 translation，下游原样复用。
    """
    _ensure_ffmpeg()
    video_path = Path(video_path)
    subs: List[dict] = []
    with tempfile.TemporaryDirectory(prefix="vs-asr-") as td:
        chunks = _extract_chunks(video_path, Path(td), chunk_seconds)
        total = len(chunks)
        n = len(keys)
        for i, (start, end, cpath) in enumerate(chunks):
            if progress:
                progress(f"AI 识别字幕中… ({i + 1}/{total})")
            # 全部可用 key 都被限流时，退避等待后重试本段（骑过免费配额的每分钟窗口），
            # 而不是直接让整个任务失败。
            raw = ki = None
            for rl_try in range(_RL_MAX_RETRY + 1):
                try:
                    raw, ki = _gemini_transcribe(cpath, keys, model,
                                                 start=_cursor_take(n), log=progress)
                    break
                except RateLimited:
                    if rl_try >= _RL_MAX_RETRY:
                        raise SubtitleError(
                            "全部有效 key 配额耗尽（HTTP 429）。请稍后等配额重置再试，"
                            "或在 gemini_keys.txt 增加更多有效 key。"
                        )
                    if progress:
                        progress(f"全部 key 暂时限流，等待 {_RL_WAIT}s 后重试第 {i + 1}/{total} 段…")
                    time.sleep(_RL_WAIT)
            _cursor_advance(ki, n)
            for s, e, body in _chunk_to_cues(_strip_fences(raw), start, end):
                body = normalize_punct(to_simplified(body))   # 繁→简 + 半角标点→全角
                if body:
                    subs.append({"start": s, "end": e,
                                 "original": body, "translation": body})
    if not subs:
        raise SubtitleError("语音识别未得到有效字幕（可能是纯音乐/无语音）")
    return subs


# ---------------------------------------------------- 混合：Whisper 时间 + Gemini 文本

_CORRECT_PROMPT = (
    "You are given an audio clip and a rough machine transcription of it, split into "
    "N numbered lines. The line TIMING/segmentation is correct, but the TEXT may have "
    "recognition errors. Listen to the audio and return the CORRECTED transcription "
    "for each line, in the ORIGINAL spoken language. If Chinese, use Simplified "
    "Chinese (简体中文). Keep EXACTLY N lines in the SAME order — line i must cover the "
    "same speech as input line i. Do NOT translate, merge, split, drop, or renumber "
    "lines. If a line is unclear, keep it close to the rough text. Output ONLY a JSON "
    "array: [{\"id\":1,\"text\":\"...\"}, ...] with exactly one object per input line.\n\n"
    "Rough transcription lines:\n"
)


def _batch_by_span(segs: List[dict], max_span: float) -> List[List[dict]]:
    """把 whisper 段按累计时长分批（每批音频跨度 ≤ max_span 秒），减少 Gemini 请求数。"""
    batches, cur, cur_start = [], [], None
    for s in segs:
        if not cur:
            cur, cur_start = [s], s["start"]
        elif s["end"] - cur_start <= max_span:
            cur.append(s)
        else:
            batches.append(cur)
            cur, cur_start = [s], s["start"]
    if cur:
        batches.append(cur)
    return batches


def _parse_correction(text: str, count: int) -> Optional[List[Optional[str]]]:
    """解析 Gemini 校正返回的 JSON 数组 → 长度 count 的文本列表（缺失项为 None）。"""
    arr = None
    try:
        arr = json.loads(text)
    except Exception:
        m = re.search(r"\[.*\]", text, re.S)
        if m:
            try:
                arr = json.loads(m.group(0))
            except Exception:
                arr = None
    if not isinstance(arr, list):
        return None
    res: List[Optional[str]] = [None] * count
    for item in arr:
        if isinstance(item, dict) and "id" in item:
            try:
                idx = int(item["id"]) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < count:
                res[idx] = str(item.get("text", "")).strip()
    return res


def transcribe_hybrid(video_path, whisper_segs: List[dict], keys: List[str],
                      model: str = DEFAULT_MODEL, batch_span: float = 60.0,
                      progress: Optional[Callable[[str], None]] = None) -> List[dict]:
    """
    混合：用 whisper_segs 的精确时间轴 + Gemini 听音频校正每段文本。
    输入 whisper_segs: [{start,end,text}]（绝对视频时间）。
    输出 [{start,end,original,translation}]：时间取 whisper，文本取 Gemini 校正
    （某批校正失败/限流/数量不符时，该批保留 whisper 原文，保证不丢句）。
    """
    _ensure_ffmpeg()
    video_path = Path(video_path)
    if not whisper_segs:
        return []
    batches = _batch_by_span(whisper_segs, batch_span)
    n = len(keys)
    out: List[dict] = []
    with tempfile.TemporaryDirectory(prefix="vs-hyb-") as td:
        for bi, batch in enumerate(batches):
            if progress:
                progress(f"Gemini 校正文本… ({bi + 1}/{len(batches)})")
            b_start = batch[0]["start"]
            b_end = batch[-1]["end"]
            slice_path = Path(td) / f"b{bi:03d}.mp3"
            _run([
                "ffmpeg", "-y", "-ss", f"{b_start:.3f}", "-t", f"{max(0.1, b_end - b_start):.3f}",
                "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
                "-b:a", "48k", str(slice_path),
            ], timeout=600)
            corrected = None
            if slice_path.exists() and slice_path.stat().st_size > 0:
                lines = "\n".join(f"{i + 1}. {seg['text'].strip()}"
                                  for i, seg in enumerate(batch))
                try:
                    raw, ki = _gemini_transcribe(
                        slice_path, keys, model, start=_cursor_take(n), log=progress,
                        prompt=_CORRECT_PROMPT + lines, max_tokens=16384)
                    _cursor_advance(ki, n)
                    corrected = _parse_correction(_strip_fences(raw), len(batch))
                except RateLimited:
                    if progress:
                        progress("Gemini 限流，本批保留 Whisper 文本")
                except Exception as e:
                    if progress:
                        progress(f"本批校正失败，保留 Whisper 文本：{e}")
            for i, seg in enumerate(batch):
                text = corrected[i] if (corrected and i < len(corrected) and corrected[i]) else seg["text"]
                text = normalize_punct(to_simplified(text))
                if text:
                    out.append({"start": seg["start"], "end": seg["end"],
                                "original": text, "translation": text})
    return out
