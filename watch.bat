@echo off
rem Двойной клик: следит за папкой inbox. Кидайте туда стримы и фильмы — нарезки появятся в results\<имя файла>\
cd /d "%~dp0"
python autoshorts.py --watch inbox -o results %*
pause
