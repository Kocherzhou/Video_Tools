# MOSS-TTS 配音进阶 — 集成方案（设计稿，待拍板）

> 决策已定：**MOSS-TTS v1.5 跑云端 AutoDL（不碰 3080）**；本稿只出设计，拍板后再写代码。
> 关联：[OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) · [RVC-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) · 飞书流程文档。

## 1. 目标 / 解决的痛点

视频里那期 AI 配音的毛病 → 本方案如何解：

| 痛点（IndexTTS/SVC） | MOSS-TTS v1.5 + 本设计 |
|---|---|
| 读得越来越快、漂移 | v1.5「stable punctuation-following prosody」+ 现有 `cursor/dub_timeline` 漂移吸收（app.py:2207 附近） |
| 不按标点断句、停顿不可控 | v1.5 显式停顿语法 `[pause X.Ys]` |
| 电音/沙沙、音质差 | 用 RVC（≈1:1 音色）替代 SVC 做参考音色；MOSS 直接克隆，不叠 SVC 失真 |
| 音色不像 | RVC 还原 + MOSS v1.5「reduced cloning variance」 |

非目标（本期不做）：把 RVC 训练/UVR5 做进 Web app。RVC 是**一次性离线**制作参考音色，见 §6 附录。

## 2. 架构总览

```
[视频字幕 app (本机 28500)]                        [云端 AutoDL GPU]
  dubbing_task / slide_dub_task                       MOSS-TTS v1.5 服务
        │  逐段文本 + 参考音色                          (Gradio :7860  或
        ├── tts_moss(text, ref_audio) ──HTTP──▶          SGLang-Omni OpenAI 兼容)
        │                              ◀──wav──
        └── concat 24k → 烧字幕 → 成片
```

- MOSS 当**无状态远程 TTS 服务**；app 只做客户端，和现有 `tts_fish`（app.py:1723，已是外部 HTTP API + reference_id 模式）一个路子。
- 参考音色（RVC 产出的「我的音色 + 饱满情绪」片段）**上传一次、长期复用**。

## 3. 后端改动（app.py）

### 3.1 新增 `tts_moss()` 适配器
位置：紧挨现有 `tts_fish`（app.py:1723）。签名对齐其它 `tts_*`：
```python
def tts_moss(text, ref_audio_path, output_path, pause_map=None):
    # 调 MOSS_BASE_URL;协议由 MOSS_PROTOCOL 切换(gradio | openai)
    # 回 wav 字节写 output_path;采样率不限(concat_wav_files 统一 24k)
```
两种协议二选一（按你部署的镜像）：
- **gradio**：`gradio_client` 或 `POST {BASE}/run/predict`，参数 [text, ref_audio, ...]。对应视频那个现成镜像。
- **openai**：`POST {BASE}/v1/audio/speech`，body 带 model/input/voice/reference。对应自建 SGLang-Omni。
> ⚠ 开放项：两种镜像的**确切请求字段**要拿到实例后核对（尤其参考音频是路径、URL 还是 base64）。适配器留一层薄封装，确定后只改一处。

### 3.2 分派加 `moss` 分支（**两处**，保持对称）
- `dubbing_task`：app.py:2159 的 `if engine == "gemini" … elif` 链里加 `elif engine == "moss": tts_moss(...)`
- `slide_dub_task`：app.py:2717 同样加一支
- 语速：MOSS 不吃 `rate/speed` 参数 → **归到「gemini 类」**（`supports_per_seg_speed = engine in ("edge","melo")` 保持不变，app.py:2196），靠 `cursor/dub_timeline` 吸收时长漂移；MOSS 本身更稳，漂移远小于 IndexTTS。进阶：紧的段用 `[pause]` 微调（二期）。

### 3.3 参考音色管理（新）
- 新目录 `voices/`（gitignore；和 uploads 一样不入库）。每个参考音色：`voices/<id>.wav` + `<id>.json`（显示名、语言、创建时间）。
- 新接口：
  - `POST /api/voices`（上传 wav → 存 id）
  - `GET  /api/voices`（列举，供前端选择器）
  - `DELETE /api/voices/<id>`
- VOICE_ROUTE（app.py:1818）**不写死** MOSS 音色；改在分派里识别 `moss:<voice_id>` 形式（现有 `elif ":" in voice` 已能拆 engine:voice，app.py:2161），`voice_id` 即 `voices/` 里的文件名。

### 3.4 配置（.env，沿用 OLLAMA_BASE_URL 风格 app.py:1062）
```
MOSS_BASE_URL=            # 云端实例地址,如 http://<autodl-ip>:7860
MOSS_PROTOCOL=gradio      # gradio | openai
MOSS_API_KEY=             # 自建 openai 兼容时用
MOSS_MODEL=MOSS-TTS-v1.5  # openai 模式选模型
```
AutoDL 实例 IP/端口**每次开机会变** → 地址放 .env 或加一个设置项（前端填一次）。改 .env 后需重启服务（沿用现状）。

### 3.5 容错
- 每段已有 `try/except → add_silence_wav`（app.py:2253）。补：MOSS 整体不可用时**回退 edge_yunjian**，并在进度里红字提示（别静默出静音）。
- 断点续跑：`if not seg_path.exists()`（app.py:2245）天然生效，MOSS 段同样缓存，中途断了重跑只补缺段——符合你一贯的中间态续跑要求。

### 3.6 采样率 / 拼接
**无需改**：`concat_wav_files`（app.py:1797）已强制重编码到 24kHz 单声道，MOSS 返回 24k/48k 都行。

## 4. 前端改动（templates/index.html）

- 配音音色选择器 + `slideVoice` 选择器加一组「🎙 MOSS 克隆（我的音色）」选项，值用 `moss:<voice_id>`。
- 参考音色区块：上传 wav（接 `/api/voices`）+ 下拉选已存音色 + 试听。
- **仅当 MOSS 就绪才显示**：`/api/check`（app.py:2931 之外的那个 check）增加 `moss_ready`（= `MOSS_BASE_URL` 已配且健康探测通过），前端据此显隐，避免没配也露出来。

## 5. 云端部署（AutoDL）

- 路线 A（最快）：用视频里的现成镜像，Gradio :7860，`MOSS_PROTOCOL=gradio`。关机不计费（视频 8:22 提到），用时开机。
- 路线 B（更稳/可并发）：自建 `SGLang-Omni` OpenAI 兼容服务，`MOSS_PROTOCOL=openai`。
- VRAM：8B PyTorch 推理约 24G（视频实测）；若想压成本，Local-Transformer-v1.5(4B) 或 llama.cpp 量化更省，但本期既然走云端，优先 8B 全量保质量。
- 安全：实例别公网裸奔——加 token，或只在内网/反代后访问（沿用你 <tunnel-host> 反代那套思路）。

## 6. 附录：一次性 RVC 参考音色制作（离线，手动/脚本清单）

照视频流程，产出 **1 个**「我的音色 + 饱满情绪」参考片段（约 1 分钟），上传进 `voices/` 即可长期复用：
1. **录音**：自己读 ~10 分钟干净人声（训练 RVC 用）。
2. **找情绪音频**：下载一段咬字标准、情绪饱满的音频 → UVR5 去伴奏 → Adobe 增强 → 截 ~1 分钟。
3. **RVC**（本机 3080 可跑，训练 ~6G、一次性、短）：训练你的音色（~500 epoch）→ 推理：把第 2 步情绪音频转成你的音色 → 得到参考片段。
4. 上传该片段到 app 的参考音色库。
> RVC 训练在 3080 上是**短时一次性**任务，和「长视频逐段本地推理」不同，掉卡风险低；真担心也可挪云端。

## 7. 落地顺序（拍板后）

1. 起云端 MOSS 实例，**确认实际 API 请求/响应字段**（解 §3.1 开放项）。
2. 后端：`tts_moss` 适配器 + 两处分派 + `/api/voices` + `.env` 配置 + `moss_ready`。
3. 前端：音色选择器 + 参考音色上传/选择 + 显隐。
4. 端到端：上传一个参考音色，跑一段 slide 配音验证音质/停顿/对齐。
5. RVC 离线制好你的正式参考音色，替换试用音色。

## 8. 待你拍板的开放问题

1. 云端镜像用**路线 A（Gradio 现成镜像）还是 B（自建 OpenAI 兼容）**？（决定 §3.1 适配器先写哪条）
2. 先接哪个功能：**slide 配音 / 主配音 dubbing / 两个都接**？（建议先 slide，链路短好验证）
3. 参考音色**先用一个固定的，还是要做完整的「多音色库」UI**？（先一个最快出效果）
4. 云端地址变更：接受**改 .env + 重启**，还是要个**前端填地址**的设置项？
