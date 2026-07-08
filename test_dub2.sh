#!/bin/bash
JOB_ID=${1:-7bba6196b7fc42b9bcb7ead148afc256}
VOICE=${2:-云扬}
UPLOADS="/mnt/c/Users/tanga/video-subtitle/uploads"

# Clean old audio dir to force re-synthesis
rm -rf "$UPLOADS/${JOB_ID}_audio"
rm -f "$UPLOADS/${JOB_ID}_dubbed.mp4"

echo "Starting dub: job=$JOB_ID voice=$VOICE"
curl -s -X POST "http://localhost:5000/api/dub/$JOB_ID" \
  -H "Content-Type: application/json" \
  -d "{\"voice\": \"$VOICE\"}" && echo " [dub started]"

echo "Waiting for dubbed.mp4..."
for i in $(seq 1 120); do
  sleep 5
  if [ -f "$UPLOADS/${JOB_ID}_dubbed.mp4" ]; then
    SIZE=$(stat -c%s "$UPLOADS/${JOB_ID}_dubbed.mp4" 2>/dev/null || echo 0)
    echo "$(date +%H:%M:%S) dubbed.mp4 exists! Size: ${SIZE} bytes"
    # Check audio streams
    echo "Audio streams:"
    ffprobe -v quiet -show_streams -select_streams a "$UPLOADS/${JOB_ID}_dubbed.mp4" 2>/dev/null | grep -E "codec_name|sample_rate|duration"
    # Check seg files generated
    SEG_COUNT=$(ls "$UPLOADS/${JOB_ID}_audio/seg_"*.wav 2>/dev/null | wc -l)
    echo "TTS segments generated: $SEG_COUNT"
    ls "$UPLOADS/${JOB_ID}_audio/" | head -10
    break
  fi
  # Show progress via audio dir
  SEG=$(ls "$UPLOADS/${JOB_ID}_audio/seg_"*.wav 2>/dev/null | wc -l)
  echo "$(date +%H:%M:%S) segs so far: $SEG"
done
