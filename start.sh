#!/bin/bash
# start.sh — 启动 video-subtitle Flask 服务
cd "$(dirname "$0")"

# 加载 .env（app.py 内 python-dotenv 也会读；这里仅供脚本内 echo 检查）
# 用 source 而非 export $(... | xargs)，先剥离行内 # 注释，避免带空格/注释的值报错
if [ -f .env ]; then
  set -a
  source <(sed -E 's/[[:space:]]+#.*$//' .env)
  set +a
fi

# faster-whisper (ctranslate2) lazily dlopen()s libcublas.so.12 / libcudnn at
# transcription time — NOT at model load. pip installed those under
# ~/.local/.../site-packages/nvidia/{cublas,cudnn}/lib, which isn't on the default
# loader path. Without this, GPU transcription dies mid-run with
# "Library libcublas.so.12 is not found" and the app silently falls back to Gemini.
# Resolve the dirs dynamically so a future nvidia-* version bump keeps working.
NV_LIB=$(/usr/bin/python3 - <<'PY'
import importlib.util as u
dirs = []
for m in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
    s = u.find_spec(m)
    if s and s.submodule_search_locations:
        dirs.append(list(s.submodule_search_locations)[0])
print(":".join(dirs))
PY
)
if [ -n "$NV_LIB" ]; then
  export LD_LIBRARY_PATH="$NV_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "🚀 启动 Video Subtitle Generator..."
echo "   GOOGLE_API_KEY: $([ -n "$GOOGLE_API_KEY" ] && echo '✓ 已设置' || echo '✗ 未设置')"
echo "   Whisper CUDA libs: $([ -n "$NV_LIB" ] && echo '✓ 已加入 LD_LIBRARY_PATH' || echo '✗ 未找到 nvidia pip 库')"
echo "   URL: http://localhost:${PORT:-28500}"
echo ""

/usr/bin/python3 app.py
