"""Самопроверка готового ролика и обложка — то, что монтажёр делает глазами перед публикацией.

check(): длина, разрешение, звук, громкость (LUFS), чёрный экран, «замёрзшая» картинка.
cover(): кадр с лицом покрупнее + заголовок крупно — обложка для TikTok/Reels/Shorts.
"""
import os, re, subprocess, tempfile


def _ff(args):
    return subprocess.run(['ffmpeg', '-hide_banner', '-nostats', *args], capture_output=True, text=True, errors='replace')


def check(path, expect_dur=None, size=(1080, 1920), target_lufs=-14.0):
    """Список замечаний (пустой — всё в порядке) и замеры."""
    issues, m = [], {}
    p = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=codec_type,width,height',
                        '-of', 'default=nw=1', path], capture_output=True, text=True)
    if p.returncode or not os.path.exists(path):
        return ['файл не читается'], m
    out = p.stdout
    d = re.search(r'duration=([0-9.]+)', out); m['dur'] = float(d.group(1)) if d else 0.0      # бывает duration=N/A
    w = re.search(r'width=(\d+)', out); h = re.search(r'height=(\d+)', out)
    if w and h and (int(w.group(1)), int(h.group(1))) != tuple(size): issues.append(f'размер {w.group(1)}x{h.group(1)} вместо {size[0]}x{size[1]}')
    if 'codec_type=audio' not in out: issues.append('нет звука')
    if expect_dur and abs(m['dur'] - expect_dur) > 1.5: issues.append(f'длина {m["dur"]:.1f} с вместо {expect_dur:.1f} с')
    r = _ff(['-i', path, '-vf', 'blackdetect=d=0.8:pix_th=0.08,freezedetect=n=0.002:d=4', '-af', 'ebur128', '-f', 'null', '-'])
    lufs = re.findall(r'I:\s+(-?[0-9.]+) LUFS', r.stderr)
    if lufs:
        m['lufs'] = float(lufs[-1])
        if abs(m['lufs'] - target_lufs) > 2.5: issues.append(f'громкость {m["lufs"]:.1f} LUFS (нужно около {target_lufs:.0f})')
    black = sum(float(x) for x in re.findall(r'black_duration:([0-9.]+)', r.stderr))
    if m.get('dur') and black > 0.25 * m['dur']: issues.append(f'чёрный экран {black:.0f} с')
    frozen = sum(float(x) for x in re.findall(r'freeze_duration: ?([0-9.]+)', r.stderr))
    st = re.findall(r'freeze_start: ?([0-9.]+)', r.stderr)
    if len(st) > len(re.findall(r'freeze_end:', r.stderr)): frozen += max(0.0, m['dur'] - float(st[-1]))   # стоит до самого конца — freezedetect конец не пишет
    if m.get('dur') and frozen > 0.4 * m['dur']: issues.append(f'картинка стоит {frozen:.0f} с (меню/пауза?)')
    m['black'], m['frozen'] = round(black, 1), round(frozen, 1)
    return issues, m


def free_times(words, t0, end, title_until=0.0, busy=()):
    """Моменты (от начала ролика), когда на экране нет субтитра: паузы между словами (как в captions.build).
    busy — другие надписи [(начало, конец, …)] во времени исходника: *СМЕЁТСЯ*, *КРИЧИТ* стоят как раз в паузах."""
    spans = sorted([(w['s'] - t0, min(w['s'] + 1.2, w['e'] + 0.6) - t0) for w in words] + [(b[0] - t0, b[1] - t0) for b in busy])
    out, prev = [], title_until
    for a, b in spans + [(end, end)]:
        if a - prev >= 0.3: out.append((prev + a) / 2)
        prev = max(prev, b)
    return [t for t in out if 0.3 < t < end - 0.3]


def _best_frame(path, dur, tmp, times=None):
    """Из кадров без субтитров (или 6 кадров первой половины) берём тот, где лицо крупнее всего (или самый резкий)."""
    try:
        import cv2
        from . import layout
        det = layout.finder()
    except Exception:
        det = None
    best, best_sc = None, -1.0
    ts = times[:: max(1, len(times) // 8)][:8] if times else [dur * (0.08 + 0.08 * k) for k in range(6)]
    for k, t in enumerate(ts):
        f = os.path.join(tmp, f'f{k}.png')
        if _ff(['-y', '-ss', f'{t:.2f}', '-i', path, '-frames:v', '1', f]).returncode or not os.path.exists(f): continue
        sc = 0.0
        if det is not None:
            img = cv2.imread(f)
            if img is not None:
                small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
                fs = det.find(small, 20)
                sc = max([fw * fh for _, _, fw, fh in fs] or [0]) + cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() * 0.01
        if sc > best_sc: best, best_sc = f, sc
    return best


def cover(path, title, out, font, dur=None, times=None):
    """Обложка: лучший кадр + заголовок крупным жёлтым шрифтом с обводкой в верхней трети."""
    from PIL import Image, ImageDraw, ImageFont
    if dur is None:
        p = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path], capture_output=True, text=True)
        dur = float(p.stdout.strip() or 10)
    with tempfile.TemporaryDirectory() as tmp:
        f = _best_frame(path, dur, tmp, times)
        if not f: return None
        with Image.open(f) as im: img = im.convert('RGB')           # файл закрыт до удаления папки (Windows)
    if title:
        d = ImageDraw.Draw(img); W, H = img.size; size = int(W * 0.105)
        words = title.upper().split(); lines = []
        while True:
            try: fnt = ImageFont.truetype(font, size)
            except OSError: fnt = ImageFont.load_default(size)       # нет шрифта — обложка всё равно будет
            lines, cur = [], ''
            for w in words:
                t = (cur + ' ' + w).strip()
                if not cur or d.textlength(t, font=fnt) <= W * 0.86: cur = t     # длинное слово — своя строка, без пустой перед ней
                else: lines.append(cur); cur = w
            lines.append(cur)
            fits = len(lines) <= 3 and max(d.textlength(ln, font=fnt) for ln in lines) <= W * 0.92   # и длинное слово не вылезает за край
            if fits or size < W * 0.06: break
            size = int(size * 0.9)
        y = int(H * 0.16)
        for ln in lines:
            tw = d.textlength(ln, font=fnt)
            d.text(((W - tw) / 2, y), ln, font=fnt, fill=(255, 221, 0), stroke_width=max(4, size // 9), stroke_fill=(0, 0, 0))
            y += int(size * 1.12)
    img.save(out, quality=92)
    return out
