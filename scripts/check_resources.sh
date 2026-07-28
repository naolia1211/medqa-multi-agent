#!/bin/bash
# check_resources.sh — Kiểm soát tài nguyên (CPU/RAM/DISK) TRƯỚC khi chạy op nặng.
# Dùng bởi skill `resource-guard` và agent `pipeline-runner`. Không phụ thuộc gì ngoài
# coreutils (free/df/nproc/awk). In bảng trạng thái; exit != 0 nếu có ngưỡng CRITICAL.
#
#   scripts/check_resources.sh                 # dùng ngưỡng mặc định
#   MIN_DISK_GB=5 MIN_RAM_MB=2000 scripts/check_resources.sh
#   scripts/check_resources.sh --profile dense # preset cho build dense (~6h, cần đĩa+RAM)
#   scripts/check_resources.sh --profile eval  # preset cho eval (CPU/RAM)
#
# Exit: 0 = OK, 1 = có CRITICAL (KHÔNG nên chạy op), 2 = lỗi công cụ.
set -u

# ── Ngưỡng mặc định (máy ~8GB theo README) ───────────────────────────────────
MIN_DISK_GB="${MIN_DISK_GB:-2}"     # đĩa trống tối thiểu (GB) trên fs của repo
MIN_RAM_MB="${MIN_RAM_MB:-1200}"    # RAM khả dụng tối thiểu (MB)
MAX_LOAD_PER_CORE="${MAX_LOAD_PER_CORE:-2.0}"   # load1 / số core tối đa
PATH_CHECK="${PATH_CHECK:-$(cd "$(dirname "$0")/.." && pwd)}"

# ── Preset theo loại việc ────────────────────────────────────────────────────
case "${1:-}" in
  --profile)
    case "${2:-}" in
      dense) MIN_DISK_GB=3;  MIN_RAM_MB=1500 ;;   # build dense 6h: index + checkpoint
      eval)  MIN_RAM_MB=1500; MAX_LOAD_PER_CORE=1.5 ;;  # eval: giữ CPU rảnh cho MedCPT
      corpus) MIN_DISK_GB=2 ;;
      *) echo "profile không rõ: ${2:-}" >&2; exit 2 ;;
    esac ;;
esac

command -v free >/dev/null && command -v df >/dev/null || { echo "thiếu free/df" >&2; exit 2; }

CORES=$(nproc 2>/dev/null || echo 1)
LOAD1=$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0)
RAM_AVAIL_MB=$(free -m | awk '/^Mem:/{print $7}')
RAM_TOTAL_MB=$(free -m | awk '/^Mem:/{print $2}')
DISK_AVAIL_GB=$(df -BG --output=avail "$PATH_CHECK" 2>/dev/null | tail -1 | tr -dc '0-9')
DISK_AVAIL_GB=${DISK_AVAIL_GB:-0}

# load tối đa cho phép = MAX_LOAD_PER_CORE * cores
MAX_LOAD=$(awk -v c="$CORES" -v p="$MAX_LOAD_PER_CORE" 'BEGIN{printf "%.2f", c*p}')

crit=0
status() {  # $1 nhãn, $2 giá trị, $3 ngưỡng, $4 ok(1/0), $5 đơn vị
  local mark="OK"; [ "$4" -eq 0 ] && { mark="CRITICAL"; crit=1; }
  printf "  %-22s %8s %-4s (ngưỡng %s)  [%s]\n" "$1" "$2" "$5" "$3" "$mark"
}

disk_ok=$([ "$DISK_AVAIL_GB" -ge "$MIN_DISK_GB" ] && echo 1 || echo 0)
ram_ok=$([ "${RAM_AVAIL_MB:-0}" -ge "$MIN_RAM_MB" ] && echo 1 || echo 0)
load_ok=$(awk -v l="$LOAD1" -v m="$MAX_LOAD" 'BEGIN{print (l<=m)?1:0}')

echo "── Resource guard ($(date '+%H:%M:%S')) · $CORES core · RAM tổng ${RAM_TOTAL_MB}MB ──"
status "DISK trống ($PATH_CHECK)" "$DISK_AVAIL_GB" "$MIN_DISK_GB" "$disk_ok" "GB"
status "RAM khả dụng"            "$RAM_AVAIL_MB"  "$MIN_RAM_MB"  "$ram_ok"  "MB"
status "Load 1 phút"            "$LOAD1"          "$MAX_LOAD"    "$load_ok" ""

# Container docker liên quan (nếu docker có)
if command -v docker >/dev/null 2>&1; then
  echo "  docker: $(docker ps --format '{{.Names}}' 2>/dev/null | tr '\n' ' ')"
fi

if [ "$crit" -ne 0 ]; then
  echo "=> CRITICAL: KHÔNG chạy op nặng cho tới khi giải phóng tài nguyên." >&2
  exit 1
fi
echo "=> OK: đủ tài nguyên."
exit 0
