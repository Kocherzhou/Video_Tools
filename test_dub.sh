#!/bin/bash
# Test dubbing with the last successful job
JOB_ID=${1:-7bba6196b7fc42b9bcb7ead148afc256}
VOICE=${2:-云扬}

echo "Starting dub: job=$JOB_ID voice=$VOICE"
curl -s -X POST "http://localhost:5000/api/dub/$JOB_ID" \
  -H "Content-Type: application/json" \
  -d "{\"voice\": \"$VOICE\"}"
echo ""

echo "Polling dub status..."
for i in $(seq 1 120); do
  RESP=$(curl -s "http://localhost:5000/api/dub_status/$JOB_ID" --max-time 5 -N 2>/dev/null | head -1 | sed 's/data: //')
  STATUS=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null)
  MSG=$(echo "$RESP"   | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('message',''))" 2>/dev/null)
  PROG=$(echo "$RESP"  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('progress',0))" 2>/dev/null)
  echo "$(date +%H:%M:%S) [${PROG}%] $STATUS — $MSG"
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "error" ]; then
    break
  fi
  sleep 5
done
echo "Done."
