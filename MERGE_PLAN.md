# 28600 + 28500 合并迁移计划（MERGE_PLAN）

> 状态：**规划稿**（2026-06-30）。本文件只描述方案，不改任何生产代码。
> 决策已锁：①**分阶段**（先合代码、后并端口）；②**Video_Tools(28500) 当家**，把 28600 的生成端并进来；③本地 beat-switch MTV 与云端结构排期 MTV **两种模式都保留**。

---

## 0. 目标与终态

**终态（北极星）**：Video_Tools 一个应用，内含
- **画面来源 / 生成模块**：本地 CPU（空镜 run_static、beat-switch 快速 MTV）+ 云端 GPU 分发（对口型 InfiniteTalk、军师结构排期、未来 Wan T2V/I2V）。pull-worker 调度成为应用内部的"把重活发去云端"子层。
- **出品模块**：转写/翻译、逐字卡拉OK、标题卡、署名、字幕烧录。

28600 作为**独立 app 消失**；云 worker 改为轮询合并后应用的端点。

**分阶段总纲**：
1. **代码统一**（不并端口、不动 worker）→ 风险最低，随时回退。
2. **路由去冲突 + 状态归并**（仍两端口）。
3. **并端口**（单进程单隧道 + worker 改指向）= 唯一不可逆 cut-over，单独排期、带回退脚本。
4. **收尾**（退役 28600 service、文档、统一温控）。

---

## 1. 现状快照（迁移基线）

### 1.1 两个 app
| | **28600** `InfiniteTalk/webapp/` | **28500** `video-subtitle/`（当家） |
|---|---|---|
| 角色 | 生成端：云 worker 拉取分发、对口型、军师结构排期 | 出品端 + 一套**本地** beat-switch MTV |
| 进程 | Flask `app.py`（~39KB），`PORT=28600` → 公网 `<tunnel-host>:<port>` | Flask `app.py`（~190KB），`PORT=28500` → 公网 `<tunnel-host>:<port>` |
| 服务 | systemd `mtv-webapp.service` | systemd `video-subtitle.service` |
| 鉴权 | `AUTH_TOKEN`（人）+ `WORKER_TOKEN`（worker） | `AUTH_TOKEN`（人，同值，见本地 .env） |
| 状态 | `jobs` 字典 + 落盘 `webapp/jobs_state.json` | `jobs` 字典（内存）+ `slide_tasks` 等 |
| 配套模块 | `plan_runner.py`、`vision_advisor.py`、`advisor_runner.py`、`refix_static*.py`、`watch_parent.py` | `mtv_processor.py`、`subtitle_asr.py`、`align_karaoke_b.py`、`suno_karaoke.py`、`gpu_guard.py`、`gpu_watch.py` |

### 1.2 路由面
**28600**：`/` `/login` · `/api/create` `/api/create_mtv` · `/api/advisor/start|status` · `/api/plan/start|status` · `/api/status/<id>` `/api/result/<id>` `/api/fleet` · `/api/worker/claim|input|progress|result`

**28500**：`/` `/login` `/karaoke` · `/api/upload` `/api/status/<id>` `/api/job_info` `/api/result/<id>` `/api/video` `/api/export` `/api/update_subtitles` `/api/import_srt` `/api/download` · `/api/dub*` `/api/burn_dub*` `/api/slide_dub*` `/api/slide/clear_clips` `/api/reprocess` · **`/api/mtv/create|status|download`（本地 beat-switch）** · `/api/karaoke/start|status|download` · `/api/gpu/status` `/api/check`

### 1.3 关键认知：28500 已有的 `/api/mtv/*` ≠ 28600 的产线
- 28500 `mtv_processor.process_mtv_job`：**纯本地** ffmpeg + librosa 节拍检测 + ThreadPool 按拍切画面 → 幻灯式 MTV。轻、快、CPU。
- 28600：**云端** InfiniteTalk 对口型 / run_static 空镜 / 军师结构排期（出过《Still That Boy》发布版的产线）。
- → 决策保留两者：**本地 beat-switch = 「快速预览版 MTV」；云端结构排期 = 「精品版 MTV」**，并存为两个模式。

---

## 2. 硬冲突点与解法（合一个进程时必须解）

| # | 冲突 | 解法（采用 Flask Blueprint 命名空间） |
|---|---|---|
| C1 | `/`、`/login` 两份 | 出品端保留在**根**（不破坏现有书签/卡拉OK链接）；生成端整体挂到蓝图 `gen`，`url_prefix="/gen"` → `/gen/`、`/gen/login` 由统一壳接管 |
| C2 | `/api/status/<id>`、`/api/result/<id>` 撞名、实现不同 | 生成端这些进 `/gen/api/...`；出品端保留 `/api/...` |
| C3 | `templates/index.html` 两份 | 出品端 index 保根；生成端模板改名 `gen_index.html` 挂 `/gen/`；新增**统一导航壳**（落地页 `/` 顶部 Tab：出品端 / 生成端） |
| C4 | 两套 `jobs` 字典 + `jobs_state.json` | 初期**各自独立存储**（蓝图内私有），job_id 都是 uuid，碰撞可忽略；UPLOAD_DIR 子目录分区（`uploads/gen/`、出品端现状）。第 2 阶段再评估是否统一为单一 job store |
| C5 | `before_request` 鉴权两套（28500 `_auth_gate` 用 hmac 常量比较；28600 含 `WORKER_TOKEN` 放行 `/api/worker/*`） | 合并为**一个 gate**：人端 `AUTH_TOKEN`（沿用 28500 的 hmac 版），worker 端 `/gen/api/worker/*` 走 `WORKER_TOKEN` 放行。两个 env 都进合并后 `.env` |
| C6 | 本地 GPU 消费者并进一个进程 | 28500 有 `gpu_guard`（Whisper/Ollama 翻译温控 + 角标），28600 有 `vision_advisor._cooldown`（军师节流）。合并 = **统一温控的机会**：军师改查同一 `gpu_guard`/`/api/gpu/status`，根治"两进程各算各的、叠加烤卡"。⚠️ 但"单次长推理盲区"仍在——军师长推理仍走 CPU |
| C7 | worker 拉取目标 = 28600 隧道 | 第 3 阶段才动：worker `HOME_BASE_URL` 改指向合并后端点；隧道 18444→新端口（详见 §5） |

---

## 3. 阶段一：代码统一（不并端口、不动 worker）— 可随时回退

**产出**：Video_Tools 仓内多出一个 `gen/`（或 `generation/`）Python 包，承载原 28600/webapp 的全部逻辑；两个 service 仍各跑各的端口；云 worker 完全不受影响。

步骤：
1. **把 28600/webapp 的 home 侧代码搬进 Video_Tools**（仅 home 调度侧，**不含** worker 侧 `cloud_worker.py`/`generate_infinitetalk.py`/`wan/`/权重——那些留在 InfiniteTalk 仓，AutoDL 实例继续 clone InfiniteTalk）：
   - `app.py`（28600）→ 改写为 Flask **Blueprint** `gen_bp`（`gen/views.py`），所有路由加 `/gen` 前缀，`@app.route`→`@gen_bp.route`。
   - `plan_runner.py` / `vision_advisor.py` / `advisor_runner.py` / `refix_static*.py` / `watch_parent.py` → `gen/` 包内。
   - `templates/index.html`（28600）→ `templates/gen_index.html`。
   - `start_fleet.sh` 等运维脚本登记进仓（仅文档/运维，不影响运行）。
2. **此阶段两 app 仍独立运行**：Video_Tools 的 `app.py` **暂不** register `gen_bp`（或用 `ENABLE_GEN=0` 关掉），生产 28600 继续跑它自己那份。即"代码进仓了但还没接线"，零生产影响。
3. **冒烟**：本地起一个 `ENABLE_GEN=1` 的临时实例（非 28500/28600 端口，如 28700），验证 `gen_bp` 路由能起、模板能渲染、`/gen/api/fleet` 返回。
4. **回退**：本阶段只是新增文件 + 一个默认关闭的开关，回退 = 不 register / 删 `gen/` 目录。

**Git**：单独 commit；push 前 `pull --rebase origin main`（双账号约定，见 memory `dual-account-workflow`）。

---

## 4. 阶段二：路由去冲突 + 统一壳 + 状态归并（仍两端口）

1. 在 Video_Tools `app.py` **register `gen_bp`（`url_prefix=/gen`）**，但仍由 28500 进程承载——此时 28500 进程**同时**提供出品端（根）和生成端（/gen），**但 worker 仍打 28600**（28600 进程继续在跑，作为 worker 的真后端，直到阶段三）。
   - ⚠️ 注意：阶段二 `/gen/api/worker/*` 在 28500 上**存在但无 worker 访问**（worker 还指向 28600）。这是有意的"双写期"：先让合并后的 /gen 在 28500 上跑通人端流程（建任务、军师、排期、看 fleet），worker 后端暂时仍是独立 28600。
2. **合并 `before_request` 鉴权**（C5）：一个 gate 同时处理人端 + worker 端。
3. **统一温控**（C6）：`vision_advisor` 改用 `gpu_guard`。
4. **统一导航壳**：根 `/` 落地页加 Tab/入口（出品端 · 生成端），或顶栏互链（参照 memory 里 28500 `/karaoke` 已做的"回链"）。
5. **状态**：评估是否把 `gen` 的 `jobs_state.json` 与出品端 job store 统一；初期可不统一（各自落盘）。
6. **验证**：28500 上 `/`（出品）、`/gen`（生成人端流程）都通；28600 仍独立服务 worker。

回退：去掉 `register_blueprint(gen_bp)` 即恢复纯出品端 28500。

---

## 5. 阶段三：并端口（单进程单隧道 + worker 改指向）= 唯一不可逆 cut-over

> 这是全程**唯一高风险、不可逆**的一步，单独挑低峰期做，带回退脚本。前置：阶段一/二在 28500 上已验证 /gen 全通。

cut-over runbook：
1. **隧道**：把公网 `<tunnel-host>:<port>`（现指 28600）改指向 28500，或让 worker 直接改用 `:18443`。二选一：
   - **A**：worker `HOME_BASE_URL` 从 `…:18444` 改成 `…:18443/gen`（路由前缀），隧道 18444 退役。
   - **B**：保留 18444 隧道但反代到 28500 的 `/gen`，worker 几乎不用改（仅 worker 端点路径加 `/gen`）。
   - 倾向 **A**（少一条隧道、终态干净）。
2. **worker 侧改动**（InfiniteTalk 仓 `cloud_worker.py` + `start_fleet.sh`）：claim/input/progress/result 的 URL 基址 + 路径前缀改成新端点；这要 **scp 到 AutoDL 实例**（实例仓旧，memory 反复强调开跑前 scp）。
3. **停 28600 service**：`mtv-webapp.service` stop+disable（先 stop 不 delete，留作回退）。
4. **验证**：开一台 AutoDL worker → 在 28500 `/gen` 建一个 mtv 任务 → 看 `journalctl -u video-subtitle` 出现 `claim→204`/领活 → 跑通一段 → 回传 result。
5. **回退**：worker `HOME_BASE_URL` 改回 18444 + 重启 `mtv-webapp.service` 即恢复旧拓扑（所以阶段三务必保留 28600 代码/服务定义到验证稳定）。

⚠️ 不可逆点：一旦隧道 18444 退役 + 28600 service 删除，回退成本骤增。**删除动作放阶段四、稳定运行数日后再做**。

---

## 6. 阶段四：收尾

- 删/归档 28600 service 与隧道（稳定数日后）。
- 文档：更新 memory（`home-server-setup` 两服务→一服务；`mtv-system-northstar` 终态达成；`dual-account-workflow` 端口/重启命令更新）。
- 统一 `/api/gpu/status` 温控角标镜像进生成端页面（memory 待办）。
- 评估 job store 彻底统一、UPLOAD_DIR 布局收敛。

---

## 7. 两种 MTV 模式如何并存（决策②落地）

合并后生成端提供两条 MTV 路径，UI 上做成两个入口/模式：
- **快速预览版**（本地）：复用 28500 现有 `/api/mtv/*` + `mtv_processor`，CPU 出片、秒级、零云钱。适合排图/选曲试看。
- **精品版**（云端）：28600 的 `/gen/api/create_mtv` + 军师结构排期 `/gen/api/plan/*` + pull-worker 分发，InfiniteTalk 对口型 / run_static 空镜。出最终发布素材。
- 两者产物都进**出品模块**（卡拉OK字幕 + 标题 + 署名）收口。

---

## 8. 风险登记 & 回退一览
| 阶段 | 主风险 | 回退 |
|---|---|---|
| 一 代码统一 | 几乎无（默认关） | 不 register / 删目录 |
| 二 去冲突+壳 | 路由/鉴权撞、温控合并回归 | 去掉 register_blueprint |
| 三 并端口 | **worker 失联、隧道错指、生产中断** | worker 改回 18444 + 重启 mtv-webapp |
| 四 收尾 | 删早了难回退 | 稳定数日后再删，保留服务定义 |

## 9. 开工前需逐阶段确认 / 待核实
1. 28600 `app.py` 内是否有**模块级全局/线程**（fleet 心跳、stale 重领看门狗 `_requeue_stale_loop`、SSE）会与 28500 现有线程/SSE 撞——阶段一搬码时要逐个核。
2. 28500 与 28600 是否有**同名工具函数/全局**（`jobs`、`build_ass`、`get_video_duration`、`UPLOAD_DIR`、`_auth_gate`）→ 蓝图封装时避免命名污染。
3. `requirements.txt` 合并（28600 轻、28500 重；torch/whisper 等只出品端要）。
4. 合并后单进程承载**全部本地 GPU 活**（Whisper + Ollama 翻译 + gemma4 军师）→ §C6 统一温控务必到位，否则叠加烤卡（3080 散热未修，见 memory）。
5. SSE/长任务并发：两端的 `_SSE_MAX_ITERS`、子进程模型（advisor_runner/plan_runner 用 `start_new_session` 脱离 flask）在单进程下的行为复核。

---

### 附：相关 memory
`mtv-system-northstar`（终态形状）· `home-server-setup`（两服务/端口/重启/温控）· `dual-account-workflow`（git 收口约定）· `gpu-compute-sourcing`（worker/隧道/scp）
