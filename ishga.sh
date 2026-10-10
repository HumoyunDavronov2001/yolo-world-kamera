#!/bin/bash
# YOLO-World kamera: 1000 sinf real vaqtda. Qo'shimcha: ./ishga.sh --imgsz 640 --conf 0.15
cd "$(dirname "$0")"
export QT_QPA_PLATFORM=xcb
exec .venv/bin/python kamera.py "$@"
