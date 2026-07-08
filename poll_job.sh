#!/bin/bash
JOB_ID=${1:-6916f3c5953b4f69b5a9ffebb53b6653}
echo "Polling job: $JOB_ID"
for i in $(seq 1 120); do
  RESP=$(curl -s "http://localhost:5000/api/job_info/$JOB_ID")
  if [ -z "$RESP" ]; then
    echo "$(date +%H:%M:%S) [waiting for server...]"
    sleep 3
    continue
  fi
  STATUS=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))")
  MSG=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('message',''))")
  PROG=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('progress',0))")
  echo "$(date +%H:%M:%S) [${PROG}%] $STATUS — $MSG"
  if [ "$STATUS" = "completed" ]; then
    echo ""
    echo "=== 字幕预览 (前3条) ==="
    echo "$RESP" | python3 -c "
import sys,json
d=json.load(sys.stdin)
subs=d.get('subtitles',[]) or d.get('subtitle_preview',[])
for s in subs[:3]:
    print(f\"  {s.get('start',0):.1f}s-{s.get('end',0):.1f}s: {s.get('translation','')}\")
print(f\"  总计: {len(d.get('subtitles',[]))} 条字幕\")
"
    exit 0
  elif [ "$STATUS" = "error" ]; then
    echo "ERROR: $MSG"
    exit 1
  fi
  sleep 5
done
echo "Timeout!"
exit 1
