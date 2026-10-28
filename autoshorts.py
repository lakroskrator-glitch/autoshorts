#!/usr/bin/env python3
"""autoshorts — нарезка длинного стрима/видео в вертикальные шорты 9:16 и сборки смешных моментов.

    python autoshorts.py stream.mp4 -n 5                         # шорты: эвристика
    python autoshorts.py stream.mp4 -n 5 --ai                    # моменты, тема и хэштеги выбирает Claude
    python autoshorts.py stream.mp4 --bg satisfying              # вебка сверху, залипательный фон снизу
    python autoshorts.py movie.mp4 --style cinema --look cinema --fx particles --punch   # кино-эдит
    python autoshorts.py stream.mp4 --compile -n 12              # «смешные моменты» одним видео
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, wave
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1'); os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error'); os.environ.setdefault('HF_HUB_VERBOSITY', 'error')
import warnings; warnings.filterwarnings('ignore', message='.*unauthenticated requests.*')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shorts import transcribe as tr, moments as mo, layout as lay, captions as cap, satisfying, look, align, funny, fetch, qa  # noqa: E402

STEP = 0.5


class UserError(Exception):
    """Понятная пользователю ошибка: печатаем текст без трейсбека."""


def run(cmd, what):
    """ffmpeg/ffprobe с понятной ошибкой вместо трейсбека."""
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        tail = r.stderr.decode('utf-8', 'replace').strip().splitlines()[-6:]
        raise UserError(f'{what}: ffmpeg завершился с ошибкой {r.returncode}\n  ' + '\n  '.join(tail))
    return r


def preflight():
    """Проверка окружения до долгой работы: ffmpeg/ffprobe, фильтр субтитров (libass), кодек x264."""
    for b in ('ffmpeg', 'ffprobe'):
        if not shutil.which(b):
            raise UserError(f'Не найден {b}. Установите ffmpeg и добавьте его в PATH (Windows: winget install ffmpeg; '
                            f'Mac: brew install ffmpeg-full; Linux: apt install ffmpeg).')
    f = subprocess.run(['ffmpeg', '-hide_banner', '-filters'], capture_output=True, text=True).stdout
    e = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True).stdout
    if not re.search(r'^\s*\S+\s+ass\s', f, re.M):
        raise UserError('Ваш ffmpeg собран без libass (фильтр ass) — субтитры рисовать нечем. '
                        'Mac: brew install ffmpeg-full и export PATH="$(brew --prefix ffmpeg-full)/bin:$PATH"; '
                        'Windows: сборка gyan.dev «full» или BtbN.')
    if 'libx264' not in e:
        raise UserError('Ваш ffmpeg собран без libx264. Поставьте полную сборку ffmpeg.')


def probe(path):
    """Размер с учётом поворота (видео с телефона), длительность, есть ли звук и сдвиг звука относительно видео."""
    if not os.path.isfile(path): raise UserError(f'Файл не найден: {path}')
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                        'stream=index,codec_type,width,height,start_time:stream_tags=rotate:stream_side_data=rotation:format=duration',
                        '-of', 'json', path], capture_output=True, text=True)
    if r.returncode != 0: raise UserError(f'Не удаётся прочитать видео: {path}\n  {r.stderr.strip()[:300]}')
    j = json.loads(r.stdout); vs = [s for s in j.get('streams', []) if s.get('codec_type') == 'video']
    aus = [s for s in j.get('streams', []) if s.get('codec_type') == 'audio']
    if not vs: raise UserError(f'В файле нет видеодорожки: {path}')
    v = vs[0]; w, h = int(v['width']), int(v['height'])
    rot = int(float(v.get('tags', {}).get('rotate', 0) or 0))
    for sd in v.get('side_data_list', []) or []:
        if 'rotation' in sd: rot = int(float(sd['rotation']))
    if abs(rot) % 180 == 90: w, h = h, w                               # ffmpeg и OpenCV сами поворачивают кадр
    dur = float(j.get('format', {}).get('duration') or 0)
    shift = 0.0
    if aus:
        try: shift = float(aus[0].get('start_time') or 0) - float(v.get('start_time') or 0)
        except ValueError: shift = 0.0
    return {'w': w, 'h': h, 'dur': dur, 'audio': bool(aus), 'shift': shift}


def mark_loud(words, loud):
    z = mo.hype_curve(loud, STEP)
    for w in words:
        a, b = int(w['s'] / STEP), int(w['e'] / STEP) + 1
        w['loud'] = bool(len(z[a:b]) and z[a:b].max() > 1.8)
    return words


def find_events(words, loud, laugh, s, e, lang='ru'):
    """Подписи без речи: *смеётся* по смеху, *кричит* по всплеску громкости — только в паузах между словами."""
    z = mo.hype_curve(loud, STEP); out = []
    lab = {'ru': ('смеётся', 'кричит'), 'en': ('laughing', 'screaming')}.get(lang, ('laughing', 'screaming'))
    i = int(s / STEP)
    while i * STEP < e:
        t = i * STEP                                                    # сетка совпадает с сеткой кривых громкости/смеха
        busy = any(w['s'] - 0.15 <= t + STEP and w['e'] + 0.15 >= t for w in words)
        kind = None
        if not busy and i < len(laugh) and laugh[i] > 2.2: kind = lab[0]
        elif not busy and i < len(z) and z[i] > 2.6: kind = lab[1]
        if kind:
            if out and out[-1][2] == kind and t - out[-1][1] <= 0.01: out[-1][1] = min(t + STEP, out[-1][0] + 1.6)
            else: out.append([t, t + STEP, kind])
        i += 1
    res = []
    for a, b, k in out:
        nxt = min([w['s'] for w in words if w['s'] >= b] or [b + 1])
        res.append((max(a, s), min(max(b, a + 0.7), nxt - 0.05), k))   # не залезаем на следующее слово
    return [r for r in res if r[1] - r[0] >= 0.3]


def ff_path(p):
    """Путь для фильтров ffmpeg без кавычек: два уровня экранирования (опция фильтра и граф). Проверено на C:, кириллице, апострофе."""
    p = p.replace('\\', '/')
    v = re.sub(r"([\\':])", r'\\\1', p)
    return re.sub(r"([\\'\[\],;])", r'\\\1', v)


def concat_line(p):
    return "file '" + p.replace("'", r"'\''") + "'\n"


def whoosh(path, sr=44100, d=0.45):
    """Короткий свуш для склеек сборки."""
    rng = np.random.default_rng(3); n = int(sr * d); t = np.linspace(0, 1, n)
    x = rng.standard_normal(n); y = np.zeros(n); f = 0.0
    for i in range(n):                                  # однополюсный фильтр с растущей частотой среза
        a = 0.02 + 0.5 * t[i] ** 2; f = f + a * (x[i] - f); y[i] = f
    y = y * t ** 1.2 * (1 - t) ** 0.6; y = y / np.abs(y).max() * 0.7
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((y * 32767).astype('<i2').tobytes())
    return path


def loud_params(video, s, d):
    """Первый проход loudnorm по звуку куска: точное попадание в −14 LUFS на коротких роликах."""
    r = subprocess.run(['ffmpeg', '-hide_banner', '-ss', f'{s:.3f}', '-t', f'{d:.3f}', '-i', video, '-vn',
                        '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json', '-f', 'null', '-'], capture_output=True, text=True)
    m = re.search(r'\{[^{}]*"input_i"[^{}]*\}', r.stderr)
    if not m: return 'loudnorm=I=-14:TP=-1.5:LRA=11'
    j = json.loads(m.group(0))
    if float(j['input_i']) < -60: return 'anull'                       # тишина — не усиливаем шум
    return (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:"
            f"measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true")


def render(video, info, clip, plan, ass, out, a, tmp, name, bg=None, flash=False, swoosh=None, punches=(), seed=0):
    s = max(0.0, clip['start'] - 0.2); d = min(clip['end'] + 0.45, info['dur']) - s
    W, H = (1920, 1080) if plan['layout'] == 'wide' else (lay.W, lay.H)
    if plan['layout'] == 'face':                       # склейки внутри куска → кадр следует за лицом в каждом плане
        plan = dict(plan, shots=lay.shot_track(video, info['w'], info['h'], s, d, plan['crop']))
    inputs = ['-ss', f'{s:.3f}', '-t', f'{d:.3f}', '-i', video]; idx = 1; bg_idx = None
    if bg and plan['layout'] == 'split':
        inputs += ['-stream_loop', '-1', '-t', f'{d:.3f}', '-i', bg]; bg_idx = idx; idx += 1
    g = lay.filter_graph(plan, bg_idx); chain = []
    if punches: chain.append(look.punch_scale([p - s for p in punches], W, H))
    if look.GRADES.get(a.look): chain.append(look.GRADES[a.look])
    if chain: g += ';[base]' + ','.join(chain) + '[base]'
    if a.fx == 'particles':
        pp = look.particles(os.path.join(tmp, f'{name}_fx.mp4'), d + 0.5, W, H, seed=seed)
        inputs += ['-i', pp]
        g += f';[base]format=gbrp[b0];[{idx}:v]format=gbrp[fx];[b0][fx]blend=all_mode=screen:all_opacity=0.55:shortest=1,format=yuv420p[base]'
        idx += 1
    g += f";[base]ass={ff_path(ass)}:fontsdir={ff_path(os.path.join(HERE, 'fonts'))}"
    if flash: g += ',fade=t=in:st=0:d=0.22:color=white'
    if a.bar and not a.compile:                         # полоска прогресса: overlay считает выражение на каждом кадре
        g += f"[vb];color=c=0xFFE100:s={W}x12:d={d:.3f}[pb];[vb][pb]overlay=x='-w+w*t/{d:.3f}':y={H - 12}:shortest=1"
    g += '[v]'
    af = f"[0:a]{loud_params(video, s, d)},aresample=48000,afade=t=out:st={max(0, d - 0.25):.3f}:d=0.25[va]"
    if swoosh:
        inputs += ['-i', swoosh]; af += f';[{idx}:a]aresample=48000,volume=0.6[w];[va][w]amix=inputs=2:duration=first:normalize=0[a]'; idx += 1
    else:
        af = af.replace('[va]', '[a]')
    run(['ffmpeg', '-v', 'error', '-y', *inputs, '-filter_complex', g + ';' + af, '-map', '[v]', '-map', '[a]', '-dn', '-sn',
         '-map_metadata', '-1', '-map_chapters', '-1', '-t', f'{d:.3f}', '-c:v', 'libx264', '-crf', '20', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-r', '30',
         '-c:a', 'aac', '-b:a', '160k', '-ar', '48000', '-ac', '2', '-movflags', '+faststart', out], f'рендер {name}')
    if os.path.getsize(out) < 10000: raise UserError(f'рендер {name}: получился пустой файл')


def caption_words(c, words, a, info):
    """Точные слова куска: распознаём с запасом ±2 с (whisper на шуме ошибается во времени на секунды),
    выравниваем по звуку и только потом отбираем слова, попавшие в момент. Звук куска декодируем отдельно (-ss) — без дрейфа."""
    s = max(0.0, c['start'] - 2.0); e = min(info['dur'], c['end'] + 2.0)
    seg = tr.load_audio(a.video, s, e - s)
    if a.caption_model != a.model:
        cw = tr.transcribe(seg, a.lang, a.caption_model, offset=s, progress=False, device=a.device, vad_threshold=0.25)
    else:
        cw = [w for w in words if s <= w['s'] <= e]
    aligned = align.align(cw, seg, s, a.lang) if a.align and cw else None
    cw = aligned if aligned is not None else tr.snap_onsets(cw, seg, offset=s)
    cw = tr.fill_and_tighten(cw, seg, s, a.lang, a.caption_model, a.device, align.align if a.align else None)
    return [w for w in cw if c['start'] - 0.25 <= w['s'] <= c['end'] + 0.2]


def check_args(a, info):
    if a.n < 1: raise UserError('-n должно быть не меньше 1')
    dmin = a.min if a.min is not None else (5 if a.compile else 15)
    dmax = a.max if a.max is not None else (25 if a.compile else 60)
    if dmin <= 0 or dmax <= 0 or dmin > dmax: raise UserError(f'Неверные --min/--max: {dmin}/{dmax} (нужно 0 < min ≤ max)')
    cam = None
    if a.cam:
        try:
            cam = tuple(int(v) for v in a.cam.split(','))
            assert len(cam) == 4
        except (ValueError, AssertionError):
            raise UserError('--cam нужно задать как x,y,w,h, например 1460,690,420,270')
        if cam[2] < 16 or cam[3] < 16: raise UserError('--cam: рамка слишком маленькая (минимум 16×16)')
        if cam[0] < 0 or cam[1] < 0 or cam[0] + cam[2] > info['w'] or cam[1] + cam[3] > info['h']:
            raise UserError(f'--cam {a.cam} выходит за кадр {info["w"]}x{info["h"]}')
    if a.bg and a.bg != 'satisfying' and not os.path.isfile(a.bg): raise UserError(f'Фон --bg не найден: {a.bg}')
    if a.ai:
        miss = mo.ai_available()
        if miss: raise UserError(miss)
    if not info['audio']: raise UserError('В видео нет звуковой дорожки, а autoshorts режет по речи.')
    return dmin, dmax, cam


VIDEO_EXT = ('.mp4', '.mkv', '.mov', '.flv', '.ts', '.webm', '.avi', '.m4v')


def watch(folder, rest, out_root, poll=20):
    """Режим папки: кладёте стрим или фильм в folder — он нарежется сам в out_root/<имя файла>/.
    Вместо видео можно положить .txt со ссылкой (и флагами после неё): https://vkvideo.ru/... --to 1h
    Файл берётся, когда перестал расти (докачался/дозаписался). Готовые помечаются, повторно не режутся."""
    import time
    os.makedirs(folder, exist_ok=True); os.makedirs(out_root, exist_ok=True)
    print(f'👀 Слежу за папкой {os.path.abspath(folder)} — кидайте туда видео. Результаты: {os.path.abspath(out_root)}. Ctrl+C — выход.')
    sizes = {}
    while True:
        for fn in sorted(os.listdir(folder)):
            p = os.path.join(folder, fn)
            link = fn.lower().endswith(('.txt', '.url'))
            if not (fn.lower().endswith(VIDEO_EXT) or link) or not os.path.isfile(p): continue
            base = os.path.splitext(fn)[0]; dst = os.path.join(out_root, base)
            if os.path.exists(os.path.join(dst, '.done')) or os.path.exists(os.path.join(dst, '.failed')): continue
            sz = os.path.getsize(p)
            if sizes.get(p) != sz: sizes[p] = sz; continue          # ещё растёт — ждём следующий круг
            print(f'\n=== {fn} ===', flush=True); os.makedirs(dst, exist_ok=True)
            src, extra = p, []
            if link:                                            # первая ссылка в файле; флаги после неё на той же строке
                txt = open(p, encoding='utf-8', errors='replace').read()
                m = re.search(r'(https?://\S+)(.*)', txt)
                if not m: print(f'  в {fn} нет ссылки — пропускаю'); open(os.path.join(dst, '.failed'), 'w').close(); continue
                src, extra = m.group(1), m.group(2).split()
            r = subprocess.run([sys.executable, os.path.abspath(__file__), src, '-o', dst, *rest, *extra])
            open(os.path.join(dst, '.done' if r.returncode == 0 else '.failed'), 'w').close()
            print(f'=== {fn}: {"готово" if r.returncode == 0 else "ошибка, см. выше"} → {dst}', flush=True)
        time.sleep(poll)


def main():
    if '--watch' in sys.argv:                           # autoshorts.py --watch inbox [-o results] [любые флаги]
        argv = sys.argv[1:]; i = argv.index('--watch'); folder = argv[i + 1]; rest = argv[:i] + argv[i + 2:]
        out_root = 'results'
        if '-o' in rest: j = rest.index('-o'); out_root = rest[j + 1]; rest = rest[:j] + rest[j + 2:]
        try: watch(folder, rest, out_root)
        except KeyboardInterrupt: print('\nОстановлено.')
        return
    for st in (sys.stdout, sys.stderr):                 # консоль Windows с cp1251/cp866 не упадёт на ▶ и ✓
        try: st.reconfigure(errors='replace')
        except Exception: pass
    ap = argparse.ArgumentParser(description='Длинное видео/стрим → вертикальные шорты или сборка смешных моментов')
    ap.add_argument('video', help='видеофайл или ссылка (VK Видео, YouTube, Twitch…); режим папки: --watch ПАПКА')
    ap.add_argument('--from', dest='t_from', help='резать с этого места: 90, 15m, 1h, 1:20:00')
    ap.add_argument('--to', dest='t_to', help='резать до этого места (со ссылкой скачивается только этот кусок): 1h, 45:00')
    ap.add_argument('--at', help='свои моменты вместо автоподбора: 37:56-38:54,2679-2736 (время в исходнике, края подтянутся к словам)')
    ap.add_argument('-o', '--out', default='out', help='папка для результатов (по умолчанию out)')
    ap.add_argument('-n', type=int, default=5, help='сколько шортов / моментов в сборке')
    ap.add_argument('--min', type=float, help='мин. длина шорта, с (по умолчанию 15; со --compile 5)')
    ap.add_argument('--max', type=float, help='макс. длина шорта, с (по умолчанию 60; со --compile 25)')
    ap.add_argument('--lang', default='ru')
    ap.add_argument('--model', default='base', help='whisper для поиска по всему видео (быстрая): tiny/base/small')
    ap.add_argument('--caption-model', default='large-v3-turbo', help='whisper для субтитров выбранных моментов (точная): large-v3-turbo, medium, small')
    ap.add_argument('--no-align', dest='align', action='store_false', help='не выравнивать слова по звуку (wav2vec2)')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='на чём считать whisper (cuda — видеокарта NVIDIA с CUDA 12 и cuDNN 9)')
    ap.add_argument('--ai', action='store_true', help='моменты, тему и хэштеги выбирает Claude (нужен ANTHROPIC_API_KEY)')
    ap.add_argument('--topic', help='о чём видео и кто в нём (имена, тема) — подсказка агентам и Claude; для ссылок берётся название видео')
    ap.add_argument('--compile', action='store_true', help='сборка смешных моментов одним видео')
    ap.add_argument('--funny', action='store_true', help='шорты из самых смешных моментов (как в сборке)')
    ap.add_argument('--quick', action='store_true', help='без агентов: моменты только по эвристике, без нейросетей смеха и смысла (быстрее)')
    ap.add_argument('--format', choices=['vertical', 'horizontal'], default='vertical', help='формат сборки')
    ap.add_argument('--layout', choices=['auto', 'split', 'face', 'fit'], default='auto')
    ap.add_argument('--cam', help='вебка вручную: x,y,w,h в пикселях исходника')
    ap.add_argument('--bg', help='фон для нижней части split: путь к видео или "satisfying"')
    ap.add_argument('--style', choices=list(cap.STYLES), default='stream', help='субтитры: ' + ', '.join(cap.STYLES))
    ap.add_argument('--look', choices=list(look.GRADES), default='none', help='цветокор: cinema, cold, warm')
    ap.add_argument('--fx', choices=['none', 'particles'], default='none', help='частицы поверх кадра')
    ap.add_argument('--punch', action='store_true', help='удары зумом на громких/акцентных словах')
    ap.add_argument('--title', action='store_true', help='заголовок-крючок на первые 3 с')
    ap.add_argument('--bar', action='store_true', help='полоска прогресса')
    a = ap.parse_args()
    tmp = None
    try:
        preflight()
        try:
            t0 = fetch.parse_time(a.t_from) or 0.0; t1 = fetch.parse_time(a.t_to); a.offset = t0
        except ValueError as e:
            raise UserError(f'Не понял время «{e}»: пишите 90, 15m, 1h, 1h30m или 1:20:00')
        if t1 is not None and t1 <= t0: raise UserError('--to должно быть больше --from')
        at = []                                        # --at разбираем сразу: ошибка в нём не должна ждать скачивания и whisper
        for part in filter(None, (p.strip() for p in (a.at or '').split(','))):
            try:
                x, y = (fetch.parse_time(v) for v in part.split('-'))
            except ValueError:
                raise UserError(f'--at: не понял «{part}» — нужно начало-конец, например 37:56-38:54')
            if y <= x: raise UserError(f'--at: в «{part}» конец не позже начала')
            if y <= t0 or (t1 is not None and x >= t1): raise UserError(f'--at: «{part}» вне куска --from/--to')
            at.append((part, x - t0, y - t0))              # время в исходнике → время в вырезанном куске
        if a.at and not at: raise UserError('--at: не задано ни одного момента, пример: 37:56-38:54,2679-2736')
        try:
            if fetch.is_url(a.video):
                a.video = fetch.fetch(a.video, os.path.join(a.out, 'source'), t0, t1)
            elif t0 or t1:
                full = probe(a.video)['dur']
                if full and t0 >= full: raise UserError(f'--from {fetch.hms(t0)} за концом видео: оно длится {fetch.hms(full)}')
                base = os.path.splitext(os.path.basename(a.video))[0]; src = os.path.join(a.out, 'source')
                os.makedirs(src, exist_ok=True)
                print(f'▶ вырезаю {fetch.hms(t0)}–{fetch.hms(t1)}…' if t1 else f'▶ вырезаю с {fetch.hms(t0)} до конца…')
                a.video = fetch.cut(a.video, os.path.join(src, f'{base}_{int(t0)}-{int(t1) if t1 else "end"}.mp4'), t0, t1)
        except fetch.FetchError as e:
            raise UserError(str(e))
        info = probe(a.video); vw, vh, dur = info['w'], info['h'], info['dur']
        dmin, dmax, cam = check_args(a, info)
        os.makedirs(a.out, exist_ok=True)
        print(f'▶ {os.path.basename(a.video)}: {vw}x{vh}, {dur / 60:.1f} мин')
        print('▶ звук, громкость, смех…'); audio = tr.load_audio(a.video); loud = tr.loudness(audio, STEP)
        laugh = mo.laugh_curve(audio, tr.SR, STEP)
        print('▶ движение в кадре (меню и паузы отсеиваются)…'); motion = tr.motion(a.video)
        print(f'▶ распознавание речи (whisper {a.model})…')
        words = tr.transcribe_cached(a.video, audio, a.lang, a.model, os.path.join(a.out, tr.cache_key(a.video, a.lang, a.model)),
                                     device=a.device, offset=info['shift'])
        print(f'  слов: {len(words)}')
        if not words: raise UserError('В видео не распознано речи — резать не по чему.')

        load = lambda s, d: tr.load_audio(a.video, max(0.0, s), d)
        def retranscribe(c):                          # точный текст момента (turbo) — для агентов и постов
            ws = tr.transcribe(load(c['start'], c['end'] - c['start']), a.lang, a.caption_model, offset=c['start'],
                               progress=False, device=a.device, vad_threshold=0.25)
            return ' '.join(w['w'] for w in ws)
        context = a.topic or fetch.last_title             # о чём видео и кто в нём: имена агенты берут отсюда, а не из расшифровки

        if at:
            clips = []
            for part, x, y in at:
                ws = [w for w in words if w['s'] >= x - 0.4 and w['e'] <= y + 0.6]
                if len(ws) < 2: raise UserError(f'--at: в {part} нет речи')
                clips.append({'start': ws[0]['s'], 'end': ws[-1]['e'], 'score': 0, 'title': mo.short_title(ws), 'hook': '',
                              'description': '', 'hashtags': [], 'emphasis': [], 'why': 'выбрано вручную (--at)',
                              'text': ' '.join(w['w'] for w in ws)})
            print(f'▶ свои моменты: {len(clips)}')
            if not a.quick:
                for c in clips: c['text'] = retranscribe(c) or c['text']
        elif a.ai:
            print('▶ Claude выбирает моменты и придумывает темы…')
            clips = mo.with_claude(words, loud, STEP, a.lang, a.n, dmin, dmax, topic=a.topic, funny=a.compile, laugh=laugh)
        else:
            fun = a.compile or a.funny; k = a.n if a.quick else a.n * 3
            if fun:
                print('▶ ищу смешные моменты (смех в звуке, «ахах», реакции)…')
                clips = mo.heuristic_funny(words, loud, laugh, STEP, a.lang, k, dmin, dmax, motion)
            else:
                print('▶ поиск моментов (эвристика: крючки, эмоции, всплески громкости)…')
                clips = mo.heuristic(words, loud, STEP, a.lang, k, dmin, dmax, motion)
                if not a.quick:                        # смешное тоже соревнуется: судьям нужны и истории, и ирония
                    fun_c = mo.heuristic_funny(words, loud, laugh, STEP, a.lang, a.n * 2, dmin, dmax, motion)
                    clips += [f for f in fun_c if all(f['end'] <= c['start'] or f['start'] >= c['end'] for c in clips)]
            if not a.quick:                            # агент-«читатель»: вся расшифровка целиком, моменты по смыслу
                print('▶ агент читает всю расшифровку, как монтажёр…')
                read = funny.read_transcript(mo.sentences(words), a.lang, dmin, dmax, 'funny' if fun else 'viral', context=context)
                if read:
                    sc = float(np.percentile([c['score'] for c in clips], 75)) if clips else 1.0
                    over = lambda x, y: max(0.0, min(x['end'], y['end']) - max(x['start'], y['start'])) / max(1e-6, min(x['end'] - x['start'], y['end'] - y['start']))
                    clips = [c for c in clips if all(over(c, r) < 0.5 for r in read)]   # границы «читателя» точнее — его вариант вместо похожего
                    for r in read: r['score'] = sc
                    clips += read; print(f'  читатель предложил {len(read)}')
            if not a.quick and clips:                  # агенты: точный текст, смех по звуку, панель судей и редактор
                if fun: clips += funny.laugh_candidates(words, laugh, STEP, load, dmin, dmax, mo.sentences, taken=clips)
                clips = funny.rank(clips, a.n, a.lang, load, retranscribe, mode='funny' if fun else 'viral', context=context,
                                   cache=os.path.join(a.out, 'funny_' + tr.cache_key(a.video, a.lang, a.caption_model)))
        if not (a.ai or a.at): clips = mo.trim_static_start(clips, words, motion, dmin)
        clips = mo.fix_bounds(clips, words, dmax, laugh=laugh, hype=mo.hype_curve(loud, STEP), step=STEP)   # контролёр: фраза целиком и развязка
        spare = [c for c in clips if c.get('reserve')]; clips = [c for c in clips if not c.get('reserve')]
        if not clips: raise UserError('Подходящих моментов не нашлось — попробуйте другие --min/--max или --ai.')
        if len(clips) < a.n and not at: print(f'  нашлось {len(clips)} из {a.n} — больше подходящих кусков в видео нет')
        if not (a.ai or a.quick or a.compile): funny.describe(clips, a.lang, context)     # описание и хэштеги к постам

        wide = a.compile and a.format == 'horizontal'
        if not wide and cam is None and a.layout in ('auto', 'split') and vw / vh > 1:
            print('▶ ищу вебку стримера…')            # оверлей вебки обычно на одном месте весь стрим — ищем один раз
            g = lay.plan(a.video, vw, vh, min(5, dur * 0.05), dur * 0.97, 'auto')
            if g['layout'] == 'split': cam = g['cam']; print(f'  вебка: x={cam[0]} y={cam[1]} {cam[2]}x{cam[3]}')
            else: print('  отдельной вебки нет — кадр подбирается для каждого момента')
        auto_cam = cam if not a.cam else None
        tmp = tempfile.mkdtemp(prefix='autoshorts_'); report = []; failed = 0
        sw = whoosh(os.path.join(tmp, 'whoosh.wav')) if a.compile else None

        queue = list(clips); i = 0
        while queue:
            c = queue.pop(0); i += 1
            name = f'{"part" if a.compile else "short"}_{i:02d}'
            out = os.path.join(tmp if a.compile else a.out, name + '.mp4')
            print(f'▶ [{i}/{len(clips)}] {c["start"]:.0f}–{c["end"]:.0f} c  «{c["title"]}»' + (' (запасной)' if c.get('reserve') else ''))
            try:
                cw = mark_loud(caption_words(c, words, a, info), loud)
                clip_cam = cam
                if auto_cam and not wide:              # сцена могла смениться (общение без игры): лицо должно быть в рамке вебки
                    fs = lay.scan_faces(a.video, vw, vh, c['start'], c['end'], n=10)
                    inside = [f for f in fs if cam[0] <= f[0] + f[2] / 2 <= cam[0] + cam[2] and cam[1] <= f[1] + f[3] / 2 <= cam[1] + cam[3]]
                    if fs and len(inside) < 0.5 * len(fs): clip_cam = None
                plan = {'layout': 'wide'} if wide else lay.plan(a.video, vw, vh, c['start'], c['end'], a.layout, clip_cam)
                bg = None
                if a.bg and plan['layout'] == 'split':
                    bg = a.bg if a.bg != 'satisfying' else satisfying.render(os.path.join(tmp, f'{name}_bg.mp4'), c['end'] - c['start'] + 1,
                                                                             seed=i, font=os.path.join(HERE, 'fonts', 'Montserrat-Black.ttf'))
                W, H = (1920, 1080) if wide else (lay.W, lay.H)
                y = int(H * 0.82) if wide else (int(H * 0.72) if a.style == 'cinema' and plan['layout'] != 'split' else lay.caption_y(plan))
                if a.compile:
                    title = (c.get('compilation_title') or 'Смешные моменты') if i == 1 else None
                else:
                    title = (c.get('hook') or c['title']) if a.title else None
                ass = os.path.join(tmp, name + '.ass')
                evs = find_events(cw, loud, laugh, c['start'], c['end'], a.lang)
                cap.build(cw, max(0.0, c['start'] - 0.2), ass, y, a.style, title, c.get('emphasis', []), W=W, H=H, events=evs)
                punches = [w['s'] for w in cw if w.get('loud')] if a.punch else []
                render(a.video, info, c, plan, ass, out, a, tmp, name, bg, flash=a.compile and i > 1,
                       swoosh=sw if a.compile and i > 1 else None, punches=punches, seed=i)
            except UserError as e:
                failed += 1; print(f'  ✗ пропущен: {e}')
                if spare: queue.append(spare.pop(0)); i -= 1; print('  контролёр: беру запасной момент')
                continue
            c.update({'file': out, 'layout': plan['layout'], 'text': ' '.join(w['w'] for w in cw)})
            mark = plan['layout']
            if not a.compile:                          # самопроверка и обложка, как перед публикацией
                try:
                    ts0 = max(0.0, c['start'] - 0.2)          # длина ровно как в render()
                    issues, m = qa.check(out, min(c['end'] + 0.45, info['dur']) - ts0)
                    c['qa'] = issues or 'ok'; c['lufs'] = m.get('lufs')
                    mark += ', проверено' if not issues else ' ⚠ ' + '; '.join(issues)
                    fatal = [x for x in issues if x.startswith(('нет звука', 'чёрный экран', 'картинка стоит'))]
                    if fatal and spare:                 # контролёр: брак не публикуем — рендерим запасной момент
                        print(f'  ✗ контролёр: {"; ".join(fatal)} — заменяю на запасной момент')
                        os.remove(out); queue.append(spare.pop(0)); i -= 1; continue
                    free = qa.free_times(cw, ts0, m.get('dur') or 0, 3.5 if a.title else 0.0, busy=evs)
                    cov = qa.cover(out, c['title'], out[:-4] + '_cover.jpg', os.path.join(HERE, 'fonts', 'Montserrat-Black.ttf'), m.get('dur'), free)
                    if cov: c['cover'] = cov
                except Exception as e:                 # проверка и обложка — подспорье: из-за них готовый ролик не теряем
                    print(f'  (самопроверка/обложка не удались: {type(e).__name__}: {str(e)[:120]})')
            report.append(c); print(f'  ✓ {mark}')
            write_report(a, report)                    # отчёт пишется по ходу — готовое не теряется при сбое
        if not report: raise UserError('Ни один ролик не собрался — см. ошибки выше.')

        if a.compile:
            lst = os.path.join(tmp, 'list.txt')
            with open(lst, 'w', encoding='utf-8') as f:
                for c in report: f.write(concat_line(c['file']))
            final = os.path.join(a.out, 'compilation.mp4')
            run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', '-movflags', '+faststart', final], 'склейка сборки')
            total = sum(c['end'] - c['start'] + 0.65 for c in report)
            W_, H_ = (1920, 1080) if wide else (lay.W, lay.H)
            try:
                issues, m = qa.check(final, None, (W_, H_))
                qa.cover(final, report[0].get('compilation_title') or 'Смешные моменты', final[:-4] + '_cover.jpg',
                         os.path.join(HERE, 'fonts', 'Montserrat-Black.ttf'), m.get('dur'))
            except Exception as e:
                issues = [f'самопроверка/обложка не удались: {type(e).__name__}']
            print(f'Готово: сборка {len(report)} моментов, ~{total:.0f} c → {final}' + (f'  ⚠ {"; ".join(issues)}' if issues else ', проверено'))
        else:
            warn = sum(1 for c in report if c.get('qa') not in (None, 'ok'))
            print(f'Готово: {plural(len(report))} в {a.out}/ (обложки — *_cover.jpg, подписи — posts.txt)'
                  + (f', пропущено {failed}' if failed else '') + (f'; замечания проверки у {warn}, см. выше' if warn else ''))
        write_report(a, report)
    except UserError as e:
        print(f'Ошибка: {e}', file=sys.stderr); sys.exit(1)
    except KeyboardInterrupt:
        print('\nОстановлено.', file=sys.stderr); sys.exit(130)
    finally:
        if tmp: shutil.rmtree(tmp, ignore_errors=True)


def plural(n):
    w = 'шорт' if n % 10 == 1 and n % 100 != 11 else 'шорта' if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else 'шортов'
    return f'{n} {w}'


def write_report(a, report):
    rep = [{k: v for k, v in c.items() if k != 'file'} | {'source_start': round(c['start'] + getattr(a, 'offset', 0.0), 2), 'file': os.path.join(a.out, 'compilation.mp4') if a.compile else c['file']} for c in report]
    with open(os.path.join(a.out, 'report.json'), 'w', encoding='utf-8') as f: json.dump(rep, f, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, 'posts.txt'), 'w', encoding='utf-8') as f:
        if a.compile and report: f.write(f"СБОРКА: {report[0].get('compilation_title') or 'Смешные моменты'}\n\n")
        for c in report:
            off = getattr(a, 'offset', 0.0)                  # при --from время — по исходному видео
            lines = [f"{fetch.hms(c['start'] + off)}–{fetch.hms(c['end'] + off)}  {c['title']}"]
            if c.get('description'): lines.append(c['description'])
            if c.get('hashtags'): lines.append(' '.join('#' + t.lstrip('#') for t in c['hashtags']))
            f.write('\n'.join(lines) + '\n\n')


if __name__ == '__main__':
    main()
