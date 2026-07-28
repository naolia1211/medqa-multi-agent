#!/bin/bash
# tg_monitor.sh — báo tiến độ 1 eval qua Telegram định kỳ, tự dừng khi đủ TOTAL.
# usage: tg_monitor.sh <LABEL> <CHECKPOINT_PATH> <TOTAL> [INTERVAL_SEC=1800]
cd /opt/code
source venv/bin/activate 2>/dev/null
LABEL="$1"; CKPT="$2"; TOTAL="$3"; INT="${4:-1800}"
while true; do
  if [ -f "$CKPT" ]; then
    DONE=$(wc -l < "$CKPT" 2>/dev/null || echo 0)
    OK=$(grep -c '"ok": true' "$CKPT" 2>/dev/null || echo 0)
    ACC=$(awk -v d="$DONE" -v o="$OK" 'BEGIN{ if(d>0){printf "%.1f",100*o/d} else {printf "0"} }')
    if [ "$DONE" -ge "$TOTAL" ]; then
      python notify.py "✅ [$LABEL] XONG: $OK/$TOTAL = ${ACC}%"
      break
    fi
    python notify.py "⏱ [$LABEL] ${DONE}/${TOTAL} · acc tạm ${ACC}%"
  else
    python notify.py "⏱ [$LABEL] đang khởi động..."
  fi
  sleep "$INT"
done
