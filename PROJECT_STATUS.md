# Video Subtitle Generator - Project Status

## 项目简介

英文视频一键生成中文字幕、配音、烧录的 Web 应用。**Whisper 本地转录 + Gemini 翻译**双引擎流水线，
字幕生成→配音→视频烧录全自动，支持多种 TTS 引擎。

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python 3.12 / Flask（单文件 `app.py`，~2000行） |
| 前端 | 单页 HTML（`templates/index.html`），原生 JS，CSS Variables 暗色主题 |
| ASR | **faster-whisper（`large-v3-turbo`）本地转录，CUDA 加速** |
| 翻译/视频理解 | Google Gemini API（`gemini-3.1-flash-lite`，含降级链） |
| TTS | Gemini TTS / Edge TTS / Fish Audio / MeloTTS / Wavenet |
| 音视频 | ffmpeg / ffprobe（编码、合并、烧录，自动检测 nvenc） |
| 字幕格式 | ASS（高级样式）、SRT |
| 运行环境 | **Windows 11 + Python 3.12 + RTX 3080 10GB（CUDA 12 + cuDNN 9）** |
| 通信 | SSE（Server-Sent Events）实时进度推送 |
| 存储 | 内存 job store + 磁盘 JSON 缓存（支持服务重启恢复） |

## 字幕生成双路径架构

```
┌─────────────────────────────────────────────────────────────────────┐
│  Whisper 路径（默认，推荐）                                          │
│  视频 → faster-whisper 本地转录（VAD，词级时间戳精度 <100ms）        │
│       → group_whisper_segments 按句子边界合并到 ~10s 块              │
│       → translate_segments_via_gemini（仅文本，无需上传视频）        │
│       → 字幕                                                         │
└─────────────────────────────────────────────────────────────────────┘
                              │ 失败时自动回退
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Gemini 路径（fallback）                                             │
│  视频 → Gemini Files API 上传 → AI 同时做 ASR+时间戳+翻译            │
│       → 短字幕合并 → silence-snap 校正时间戳                         │
│       → 字幕                                                         │
└─────────────────────────────────────────────────────────────────────┘
```

回退触发条件：`faster-whisper` 未安装 / CUDA 失败 / Whisper 返回空 / 任何异常。  
两条路径产出的字幕格式完全一致：`[{start, end, original, translation}, ...]`。

**性能对比**（366s 视频，3080）：

| 指标 | Whisper 路径 | Gemini 路径 |
|---|---|---|
| 总耗时 | **~25s** | 3-10 分钟 |
| 时间戳精度 | <100ms（VAD 级） | ±0.3-1.5s + silence-snap 校正 |
| Gemini token 消耗 | 几千字翻译 | 视频 token 上万 |
| 配额风险 | 低 | 高（free-tier 容易触发 429） |

## 功能模块

### 1. 字幕生成
- **Whisper 路径**：本地 `large-v3-turbo` + VAD filter；结尾幻觉过滤（`info.duration` clamp）；
  `condition_on_previous_text=False` 减少循环幻觉
- **Gemini 路径**：动态 Prompt（视频时长 / 10 计算目标条数），三层 JSON 解析容错（标准 → 截断恢复 → 正则），
  iterative merge（<6s 合并），ffmpeg silencedetect snap（±1.5s tolerance）

### 2. TTS 配音引擎

| 引擎 | 类型 | 声音选项 | 语速控制 | 特点 |
|------|------|----------|----------|------|
| **Gemini TTS** | 云端 | Kore(女)、Charon/Fenrir/Puck(男) | 探测式预计算（合成首句测实际语速） | 音质最好，有 rate limit（6s间隔）|
| **Edge TTS** | 云端免费 | 晓晓/晓伊/云希/云健/云扬/晓臻 | `rate="+N%"` 参数 | 微软免费，速度快 |
| **Fish Audio** | 云端 | 成熟男声 | 无 | 需 FISH_API_KEY |
| **MeloTTS** | 本地 | 中文女/男、中英混合、英文女/男 | `speed` 参数 | 需 CUDA，离线可用 |
| **Wavenet** | 云端 | 女声/男声 | 无 | 需 Google Cloud API |

**默认声音**：Edge 云扬（`zh-CN-YunyangNeural`，新闻男声，免费）

### 3. 语速预计算（统一逻辑）
- `calculate_global_speed(subtitles, chars_per_sec, max_speed=1.3)`
- Edge TTS 基准：5.5 字/秒；MeloTTS 基准：4.5 字/秒
- Gemini TTS：实测探测（合成首句 → 测量真实 chars/sec）
- 全局语速 = min(max_ratio * 1.05, 1.3)
- 写入 buffer 时只补静音，不截断音频

### 4. 字幕烧录
- ASS 格式，Noto Sans CJK SC 字体
- 前端可调：字体大小、每行字数、描边粗细、字幕颜色、位置
- 参数保存到 localStorage，烧录时传给后端
- GPU 加速编码（h264_nvenc 自动检测）

### 5. Slide Deck 配音
- 上传 PPT/PDF → 转 JPEG（LibreOffice + pdftoppm）
- Gemini Vision 生成中文讲解词（50-80字/页）
- 逐页 TTS 合成 → 页间 0.8s 静音 → 输出 MP3
- 关联视频字幕作为讲解主题
- **断点续传**：图片/讲解词/TTS 音频三阶段均缓存

### 6. 前端功能
- 拖拽上传 + 4步进度条 + SSE 实时日志
- 视频内嵌字幕预览（可拖拽定位 / 拖拽缩放字号）
- 字幕位置预设：左上/右上/居中/左下/右下
- 字幕样式面板：字体大小、每行字数、描边粗细、颜色（含实时预览）
- 历史记录：localStorage 存储，中文标题，**显示字幕来源角标（🎯 Whisper / ☁ Gemini）**
- 导出 SRT / 下载原始视频 / 下载字幕视频 / 下载配音视频 / 下载字幕+配音视频

## 模型与降级链

```python
# app.py 顶部
MODEL_MAP = {
    "gemini-3.1-flash":      "gemini-3.1-flash-lite",  # GA 替代已下架的 -preview
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
    "gemini-3.1-pro":        "gemini-3.1-pro-preview", # free-tier limit=0，会自动 fallback
    "gemini-3.5-flash":      "gemini-3.5-flash",       # ⭐ 当前默认（I/O 2026 GA）
    # ...
}

SUBTITLE_FALLBACK_CHAIN = [
    "gemini-3.1-flash-lite",   # 首选 fallback — 免费 tier 配额最宽松
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
]

GEMINI_THINKING_LEVEL = "low"  # env-tunable: minimal/low/medium/high
```

**Gemini 3.5 Flash 简介**（I/O 2026 5/19 GA，当前默认）：
- Model ID：`gemini-3.5-flash`（无 preview 后缀）
- 性能：超越 Gemini 3.1 Pro（agent / coding 评测）；**翻译质量明显优于 flash-lite**
- 免费 tier：15 RPM / 1500 RPD（个人使用富余）
- 付费 tier：$1.50/M in, $9.00/M out
- Thinking：dynamic on by default（MEDIUM）→ 本项目设为 **LOW**，翻译任务平衡速度与质量
- Fallback：如遇限流自动降级到 `gemini-3.1-flash-lite`（更宽配额）
- **下月（2026 年 6 月）**：`gemini-3.5-pro` GA，届时再评估

`call_gemini_generate` 统一 wrapper：
- `429 + retryDelay` → sleep 后重试同模型
- `429 + limit:0` / `404 NOT_FOUND` → 切下一个模型
- SSE 日志推送降级原因（前端 warning 级别）

## 关键参数配置

```
字幕生成
├── Whisper 模型：large-v3-turbo（首次下载 1.62GB 到 ~/.cache/huggingface）
├── Whisper VAD min_silence：300ms
├── condition_on_previous_text：False（减少幻觉）
├── 末尾幻觉过滤：seg.start >= info.duration 直接 break
├── 字幕密度（Gemini fallback）：每条约 10 秒
├── 字数控制：中文 ≤ 时间槽秒数 × 4.5 字
├── 短字幕合并阈值：< 6 秒
├── silence-snap tolerance：1.5 秒（仅 Gemini 路径）
└── max_output_tokens：65536

TTS 配音
├── 全局语速上限：1.3x
├── 采样率：24000 Hz（16-bit mono）
├── Edge TTS 基准：5.5 字/秒
├── MeloTTS 基准：4.5 字/秒
├── Gemini TTS rate limit：6 秒间隔
└── TTS 模型：gemini-3.1-flash-live-preview-tts

ASS 字幕样式
├── 字体：Noto Sans CJK SC
├── 字体大小：font_size_px × 1.5（前端可调）
├── 每行最大字数：45（前端可调）
├── 描边（Outline）：3
├── 阴影（Shadow）：1.5
├── MarginV：5
├── 默认颜色：&H0000FFFF（前端可调）
└── 标点悬挂处理：自动吸收行首标点

Slide 配音
├── 图片格式：JPEG，150 DPI
├── 讲解词：50-80 字/页
├── 页间静音：0.8 秒
├── Vision 模型：gemini-2.5-flash-preview-05-20
└── 输出格式：MP3（libmp3lame -q:a 2）
```

## 环境变量

```bash
# 必须
GOOGLE_API_KEY="your-gemini-api-key"

# 可选 — Whisper（启用 / 模型 / 设备 / 计算精度）
WHISPER_ENABLED=1                       # 默认 1，设 0 强制走 Gemini 路径
WHISPER_MODEL_SIZE=large-v3-turbo       # tiny/base/small/medium/large-v3-turbo/large-v3
WHISPER_DEVICE=cuda                     # cuda / cpu
WHISPER_COMPUTE_TYPE=float16            # float16 / int8（低显存）
WHISPER_PREWARM=1                       # 启动时预加载模型，避免首请求等待

# 可选 — 其他
FISH_API_KEY="..."                      # Fish Audio TTS
AUTH_TOKEN="..."                        # API 鉴权
HF_HUB_DISABLE_SYMLINKS_WARNING=1       # 消 HuggingFace 符号链接 warning
```

`.env` 文件放项目根目录，dotenv 自动加载。

## 环境依赖

### Windows 11（当前生产环境）

```powershell
# Python 依赖
pip install flask>=3.0.0 google-genai>=1.0.0 httpx>=0.27.0 pdf2image>=1.16.0

# Whisper 本地转录
pip install faster-whisper
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"

# 系统依赖
# ffmpeg: 下载 https://www.gyan.dev/ffmpeg/builds/ 加入 PATH
# poppler: 下载 https://github.com/oschwartz10612/poppler-windows
# LibreOffice: 下载 https://www.libreoffice.org/

# 可选：MeloTTS（本地 TTS）
# pip install melo-tts

# 可选：Edge TTS（免费云端）
pip install edge-tts
```

**Windows CUDA DLL 自动发现**：`app.py` 启动时 `_setup_windows_cuda_dll_paths()` 自动把
`<site-packages>/nvidia/cublas/bin` 和 `<site-packages>/nvidia/cudnn/bin` 同时
`os.add_dll_directory()` + 前置 `PATH`（ctranslate2 用老式 LoadLibrary，必须改 PATH）。

### Linux / WSL（也支持）

```bash
sudo apt install ffmpeg poppler-utils libreoffice -y
pip install faster-whisper nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```

WSL 上 `os.add_dll_directory` 调用是 no-op，nvidia 包的 `lib/` 目录自动通过 RPATH 找到。

## 启动方式

```powershell
cd C:\Users\tanga\video-subtitle
# .env 已配好的情况下
python app.py
# 服务启动在 http://localhost:28500
```

启动日志关键行：
```
GPU:           NVIDIA GeForce RTX 3080 (driver xxx) ✓ nvenc 可用
Whisper:       preloading large-v3-turbo on cuda...  ← WHISPER_PREWARM=1 时
Whisper:       ready (large-v3-turbo on cuda)
```

## 已知问题 / 待优化

1. **服务重启后 job 恢复有限**：内存中 job 状态丢失，磁盘 JSON 缓存仅恢复字幕数据，配音状态无法恢复
2. **Gemini TTS rate limit**：固定 sleep(6) 较保守，高峰期仍可能被限流
3. **Slide 配音字幕位置**：burn_dub 流程中 Slide 配音使用固定居中底部位置，未关联前端字幕位置设置
4. **大文件上传**：无分片上传，超大视频可能超时
5. **并发限制**：单线程 TTS 合成，多用户同时配音会排队
6. **多家 LLM 选项未接通**：前端模型选项的 OpenAI / Claude / Qwen-VL / GLM-4V 当前仍走 Gemini client，会失败

## 文件结构

```
video-subtitle/
├── app.py                       # 后端主文件
├── templates/
│   └── index.html               # 前端单页应用
├── test_whisper.py              # Whisper GPU 诊断脚本
├── requirements.txt
├── .env                         # 环境变量（gitignore）
├── uploads/                     # 上传文件 + 中间产物（自动创建）
│   ├── {job_id}.mp4
│   ├── {job_id}_subtitles.json
│   ├── {job_id}_audio/
│   ├── {job_id}_dubbed.mp4
│   └── {task_id}_slides/
│       ├── input.pdf/.pptx
│       ├── page-*.jpg
│       ├── narrations.json
│       └── page_*_raw.wav
└── PROJECT_STATUS.md            # 本文件
```

## Whisper 故障诊断速查

如果 Whisper 路径异常（卡住 / 慢得离谱 / DLL 错误），运行：

```powershell
python test_whisper.py uploads\<某个视频文件>.mp4
```

期望输出：
- `[+] added nvidia.cublas DLL dir: ...` + `sample DLLs: [...]`
- `[+] prepended 2 dirs to PATH`
- `loaded in 1-4s`
- `total: X.Xs for Y.Ys video → 25-50x realtime`

低于 10x = 跑在 CPU 上；DLL 错误 = nvidia wheels 没装好。
