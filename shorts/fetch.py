"""Видео по ссылке (VK Видео, YouTube, Twitch, Rutube… — всё, что умеет yt-dlp) и вырезка куска по времени.

Если нужен только кусок длинного стрима (--from/--to), из HLS-потока скачиваются только нужные сегменты,
а не весь эфир: три часа стрима ради первого часа не качаем.
"""
import os, re, subprocess, sys, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor


last_title = None


class FetchError(Exception):
    pass


def is_url(s):
    return bool(re.match(r'https?://', s.strip(), re.I))


def parse_time(s):
    """«90», «90s», «15m», «1h», «1h30m», «1:20:00», «20:00» → секунды."""
    if s is None: return None
    s = s.strip().lower()
    if re.fullmatch(r'\d+(\.\d+)?', s): return float(s)
    if ':' in s:
        parts = s.split(':')
        if len(parts) > 3 or not all(re.fullmatch(r'\d+(\.\d+)?', p) for p in parts): raise ValueError(s)
        return sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))
    m = re.fullmatch(r'(?:(\d+(?:\.\d+)?)h)?\s*(?:(\d+(?:\.\d+)?)m(?:in)?)?\s*(?:(\d+(?:\.\d+)?)s)?', s)
    if not m or not any(m.groups()): raise ValueError(s)
    h, mi, se = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + se


def hms(t):
    """Секунды → «1:05:30» / «5:30»."""
    t = int(t or 0); h, m, s = t // 3600, t % 3600 // 60, t % 60
    return f'{h}:{m:02d}:{s:02d}' if h else f'{m}:{s:02d}'


def _log(msg): print(msg, file=sys.stderr, flush=True)


def _get(url, headers, tries=5):
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                return r.read()
        except Exception as e:
            last = e
    raise FetchError(f'не удалось скачать {url[:80]}…: {last}')


def _hls_range(m3u8, headers, t0, t1, out):
    """Сегменты HLS, покрывающие [t0, t1], склеиваются в один файл. None — поток зашифрован, на байтовых диапазонах или это не медиа-плейлист."""
    text = _get(m3u8, headers).decode('utf-8', 'replace')
    if '#EXT-X-STREAM-INF' in text:                                 # мастер-плейлист — берём вариант с наибольшим битрейтом
        lines, var, bw = text.splitlines(), None, -1
        for i, l in enumerate(lines[:-1]):
            if l.startswith('#EXT-X-STREAM-INF') and not lines[i + 1].startswith('#'):
                b = re.search(r'BANDWIDTH=(\d+)', l); b = int(b.group(1)) if b else 0
                if b > bw: var, bw = lines[i + 1].strip(), b
        if not var: return None
        m3u8 = urllib.parse.urljoin(m3u8, var); text = _get(m3u8, headers).decode('utf-8', 'replace')
    if re.search(r'#EXT-X-KEY:METHOD=(?!NONE)', text) or '#EXT-X-BYTERANGE' in text: return None   # шифр или куски одного файла
    init, segs, t, dur = None, [], 0.0, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('#EXT-X-MAP:'):
            u = re.search(r'URI="([^"]+)"', line)
            if not u: return None
            init = urllib.parse.urljoin(m3u8, u.group(1))
        elif line.startswith('#EXTINF:'):
            try: dur = float(line[8:].split(',')[0])
            except ValueError: return None
        elif line and not line.startswith('#') and dur is not None:
            if t + dur > t0 and t < t1: segs.append((t, urllib.parse.urljoin(m3u8, line)))
            t += dur; dur = None
    if not segs: return None
    first = segs[0][0]; part = out + '.ts'
    _log(f'  скачиваю {len(segs)} сегментов ({(segs[-1][0] - first) / 60:.0f} мин)…')
    ex = ThreadPoolExecutor(8)
    try:
        with open(part, 'wb') as f:
            if init: f.write(_get(init, headers))
            for k, data in enumerate(ex.map(lambda s: _get(s[1], headers), segs)):
                f.write(data)
                if k % 50 == 0: print(f'\r  скачано: {k * 100 // len(segs)}%   ', end='', file=sys.stderr, flush=True)
        print('\r  скачано: 100%   ', file=sys.stderr)
        # точная обрезка по времени без перекодирования (до ближайшего ключевого кадра)
        r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{max(0.0, t0 - first):.3f}', '-i', part, '-t', f'{t1 - t0:.3f}',
                            '-c', 'copy', '-movflags', '+faststart', '-f', 'mp4', out], capture_output=True, text=True)
    finally:                                                       # сбой или Ctrl+C — не докачиваем остальные сегменты
        ex.shutdown(wait=False, cancel_futures=True)
        if os.path.exists(part): os.remove(part)
    if r.returncode: raise FetchError('ffmpeg не смог собрать скачанные сегменты: ' + r.stderr.strip()[-300:])
    return out


class _Mute:
    def debug(self, msg): pass
    warning = error = info = debug


def fetch(url, folder, t0=0.0, t1=None, max_height=1080):
    """Скачивает видео (или его кусок [t0, t1]) в folder, возвращает путь. Повторный запуск берёт уже скачанное."""
    try:
        import yt_dlp
    except ImportError:
        raise FetchError('Для ссылок нужен yt-dlp: pip install yt-dlp')
    os.makedirs(folder, exist_ok=True)
    opts = {'quiet': True, 'no_warnings': True, 'noprogress': True, 'noplaylist': True,
            'format': (f'best[height<={max_height}][protocol^=m3u8]/best[height<={max_height}][acodec!=none][vcodec!=none]/'
                       f'bestvideo[height<={max_height}]+bestaudio/best')}
    _log('▶ ссылка: получаю сведения о видео…')
    try:
        with yt_dlp.YoutubeDL({**opts, 'logger': _Mute()}) as ydl:      # ошибку покажем сами, одной строкой
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        raise FetchError(f'yt-dlp не открыл ссылку: {str(e)[:300]}')
    if info.get('_type') == 'playlist': raise FetchError('Это плейлист — дайте ссылку на одно видео.')
    if info.get('is_live'): raise FetchError('Это идущий прямой эфир — дождитесь записи и дайте ссылку на неё.')
    total = info.get('duration') or 0
    if total and t0 >= total: raise FetchError(f'--from {hms(t0)} за концом видео: оно длится {hms(total)}')
    t1 = min(t1, total) if (t1 and total) else t1
    tag = re.sub(r'[^\w-]', '_', str(info.get('id') or 'video'))[:40]
    span = f'_{int(t0)}-{int(t1) if t1 else "end"}' if (t0 or t1) else ''
    out = os.path.join(folder, f'source_{tag}{span}.mp4')
    if os.path.exists(out) and os.path.getsize(out) > 0:
        _log(f'  уже скачано: {out}'); return out
    title = info.get('title') or tag
    global last_title; last_title = title                 # название видео — подсказка агентам (имена, тема)
    _log(f'  «{title[:70]}»' + (f', {hms(total)}' if total else '') + (f'; беру {hms(t0)}–' + (hms(t1) if t1 else 'конец') if span else ''))
    fmts = info.get('requested_formats') or [info]
    if (t0 or t1) and all(str(f.get('protocol', '')).startswith('m3u8') for f in fmts):
        parts = [f'{out}.{k}.mp4' for k in range(len(fmts))]
        try:
            if all(_hls_range(f['url'], f.get('http_headers') or {}, t0, t1 or total or 1e12, p) for f, p in zip(fmts, parts)):
                if len(parts) == 1: os.replace(parts[0], out); return out
                cmd = ['ffmpeg', '-v', 'error', '-y'] + sum((['-i', p] for p in parts), []) + \
                      sum((['-map', f'{k}'] for k in range(len(parts))), []) + ['-c', 'copy', '-movflags', '+faststart', '-f', 'mp4', out + '.part']
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode == 0: os.replace(out + '.part', out); return out
                _log('  (не удалось склеить видео и звук: ' + r.stderr.strip()[-200:] + ')')
        except Exception as e:                                    # битый плейлист, сеть — не беда, качаем целиком
            _log(f'  (частичная загрузка не удалась: {e}; качаю целиком)')
        finally:
            for p in parts + [out + '.part']:
                if os.path.exists(p): os.remove(p)
    full = os.path.join(folder, f'source_{tag}_full.%(ext)s')
    _log('  скачиваю видео целиком…')
    try:
        with yt_dlp.YoutubeDL({**opts, 'quiet': False, 'noprogress': False, 'outtmpl': full, 'merge_output_format': 'mp4',
                               'retries': 20, 'fragment_retries': 50}) as ydl:
            got = ydl.extract_info(url, download=True)
    except Exception as e:
        raise FetchError(f'yt-dlp не скачал видео: {str(e)[:300]}')
    full = next((d['filepath'] for d in (got or {}).get('requested_downloads') or [] if os.path.isfile(d.get('filepath') or '')), None) or \
        next((os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(f'source_{tag}_full.') and not f.endswith(('.part', '.ytdl'))), None)
    if not full: raise FetchError('yt-dlp не оставил скачанного файла')
    if not (t0 or t1): os.replace(full, out); return out
    return cut(full, out, t0, t1)


def cut(src, out, t0=0.0, t1=None):
    """Кусок локального файла без перекодирования."""
    if os.path.exists(out) and os.path.getsize(out) > 0: return out
    cmd = ['ffmpeg', '-v', 'error', '-y', '-ss', f'{t0:.3f}', '-i', src]
    if t1: cmd += ['-t', f'{t1 - t0:.3f}']
    part = out + '.part'                                           # через временный файл: оборванный запуск не оставит битый кусок
    r = subprocess.run(cmd + ['-map', '0:v:0', '-map', '0:a:0?', '-c', 'copy', '-movflags', '+faststart', '-f', 'mp4', part], capture_output=True, text=True)
    if r.returncode:
        if os.path.exists(part): os.remove(part)
        raise FetchError('ffmpeg не смог вырезать кусок: ' + r.stderr.strip()[-300:])
    os.replace(part, out)
    return out
