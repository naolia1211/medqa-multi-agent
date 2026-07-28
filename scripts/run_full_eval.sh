#!/bin/bash
cd /home/be
source /home/be/venv/bin/activate
exec python eval.py --fresh --limit 0 --split MedQA-USMLE/questions/US/test.jsonl --concurrency 6
