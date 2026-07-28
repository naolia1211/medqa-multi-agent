#!/bin/bash
# RESUME full test eval — KHÔNG có --fresh, tự bỏ qua câu đã xong trong checkpoint.
cd /home/be
source /home/be/venv/bin/activate
exec python eval.py --limit 0 --split MedQA-USMLE/questions/US/test.jsonl --concurrency 3
