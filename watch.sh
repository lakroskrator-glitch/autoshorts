#!/bin/sh
# Следит за папкой inbox: кидайте туда стримы и фильмы — нарезки появятся в results/<имя файла>/
cd "$(dirname "$0")"
exec python3 autoshorts.py --watch inbox -o results "$@"
