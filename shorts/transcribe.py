"""Звук и речь: загрузка аудио, громкость, распознавание слов с таймингами (faster-whisper), движение в кадре."""
import hashlib, json, os, re, subprocess, sys, time
import numpy as np

SR = 16000
CHUNK = 600            # длинное видео распознаём кусками по 10 мин: память не растёт с длиной
OVERLAP = 4            # перекрытие кусков, чтобы не резать слово на стыке


def load_audio(path, start=None, dur=None, has_audio=True, length=None):
    """Моно 16 кГц float32. Без аудиодорожки — тишина нужной длины (видео без звука не роняет программу)."""
    if not has_audio:
        return np.zeros(int((dur if dur is not None else (length or 0)) * SR), np.float32)
    cmd = ['ffmpeg', '-v', 'error']
    if start is not None: cmd += ['-ss', f'{start:.3f}']
    cmd += ['-i', path]
    if dur is not None: cmd += ['-t', f'{dur:.3f}']
    cmd += ['-vn', '-ac', '1', '-ar', str(SR), '-f', 's16le', '-']
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    a = np.frombuffer(raw, '<i2').astype(np.float32)               # одна копия, деление на месте
    a *= 1 / 32768
    return a


def loudness(audio, step=0.5):
    """Громкость (дБ) окнами по step секунд — по ней ищем крики, смех, «хайп». Считается блоками без копий всего аудио."""
    n = int(SR * step); k = len(audio) // n
    if k == 0: return np.array([-60.0])
    out = np.empty(k, np.float32); blk = 2000
    for i in range(0, k, blk):
        j = min(k, i + blk); x = audio[i * n:j * n].reshape(j - i, n)
        out[i:j] = np.sqrt(np.einsum('ij,ij->i', x, x) / n + 1e-10)
    return 20 * np.log10(out)


_models = {}


def _quiet_hf():
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
    import logging, warnings
    logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore', message='.*unauthenticated requests.*')


SIZES = {'tiny': '75 МБ', 'base': '145 МБ', 'small': '480 МБ', 'medium': '1.5 ГБ', 'large-v3': '3 ГБ', 'large-v3-turbo': '1.6 ГБ', 'turbo': '1.6 ГБ'}


def _model(name, device='cpu'):
    if name not in _models:
        _quiet_hf()
        from faster_whisper import WhisperModel
        try:
            from faster_whisper.utils import download_model
            download_model(name, local_files_only=True)
        except Exception:
            print(f'  скачиваю модель whisper «{name}» (~{SIZES.get(name, "?")}, один раз)…', file=sys.stderr, flush=True)
        try:
            _models[name] = WhisperModel(name, device=device, compute_type='int8' if device == 'cpu' else 'int8_float16')
        except Exception as e:
            raise SystemExit(f'Не удалось загрузить модель whisper «{name}»: {e}\n'
                             f'Проверьте интернет (нужен при первом запуске) и что faster-whisper>=1.1.')
    return _models[name]


HALLUCINATIONS = ('продолжение следует', 'спасибо за просмотр', 'редактор субтитров', 'субтитры сделал', 'субтитры создавал', 'thanks for watching')


def drop_hallucinations(words):
    """Известные «галлюцинации» whisper на музыке/тишине — выбрасываем."""
    text = ' '.join(re.sub(r'[^\w ]', '', w['w'].lower()) for w in words)
    if not any(h in text for h in HALLUCINATIONS): return words
    out, i = [], 0
    while i < len(words):
        hit = None
        for h in HALLUCINATIONS:
            n = len(h.split()); cand = ' '.join(re.sub(r'[^\w ]', '', w['w'].lower()) for w in words[i:i + n])
            if cand.startswith(h): hit = n; break
        if hit: i += hit; continue
        out.append(words[i]); i += 1
    return out


def transcribe(audio, lang, model, offset=0.0, progress=True, device='cpu', vad_threshold=0.5):
    """Слова [{'w','s','e'}] со временем относительно начала видео (offset — где начинается audio). Длинное — кусками."""
    m = _model(model, device); total = len(audio) / SR; words = []; t0 = time.time(); last = 0
    starts = list(np.arange(0, max(total, 1e-3), CHUNK)) if total > CHUNK + OVERLAP else [0.0]
    for ci, cs in enumerate(starts):
        a, b = int(cs * SR), int(min(total, cs + CHUNK + OVERLAP) * SR)
        segs, _ = m.transcribe(audio[a:b], language=lang, word_timestamps=True, vad_filter=True, condition_on_previous_text=False,
                               vad_parameters={'threshold': vad_threshold, 'min_silence_duration_ms': 400})
        keep_from = cs if ci else -1; keep_to = cs + CHUNK if ci + 1 < len(starts) else 1e12
        for s in segs:
            for w in s.words or []:
                t = cs + w.start
                if w.word.strip() and keep_from <= t < keep_to:
                    words.append({'w': w.word.strip(), 's': round(t + offset, 3), 'e': round(cs + w.end + offset, 3)})
            if progress and time.time() - last > 5:
                last = time.time(); el = last - t0; done = min(1.0, (cs + s.end) / max(total, 1e-6))
                eta = el / max(done, 1e-6) - el
                print(f'\r  распознавание: {done * 100:5.1f}%  осталось ~{eta / 60:.0f} мин   ', end='', file=sys.stderr, flush=True)
    if progress: print('\r  распознавание: 100%' + ' ' * 30, file=sys.stderr)
    return drop_hallucinations(words)


def cache_key(video, lang, model):
    """Кэш расшифровки привязан к конкретному файлу (путь, размер, время изменения), языку и модели."""
    st = os.stat(video)
    h = hashlib.sha1(f'{os.path.abspath(video)}|{st.st_size}|{int(st.st_mtime)}|{lang}|{model}'.encode()).hexdigest()[:12]
    return f'words_{model}_{lang}_{h}.json'


def transcribe_cached(video, audio, lang, model, cache, device='cpu', offset=0.0):
    if os.path.exists(cache):
        return json.load(open(cache, encoding='utf-8'))
    words = transcribe(audio, lang, model, offset=offset, device=device)
    tmp = cache + '.part'
    with open(tmp, 'w', encoding='utf-8') as f: json.dump(words, f, ensure_ascii=False)
    os.replace(tmp, cache)                                     # атомарно: оборванный запуск не оставит битый кэш
    return words


def snap_onsets(words, audio, offset=0.0):
    """Подтягивает начало слова к реальному началу звука после паузы (запасной вариант, если нет wav2vec2)."""
    hop = SR // 100; n = len(audio) // hop
    if n < 10: return words
    env = np.sqrt(np.mean(audio[:n * hop].reshape(n, hop) ** 2, axis=1))
    for i, w in enumerate(words):
        gap = w['s'] - words[i - 1]['e'] if i else 1.0
        if gap < 0.12: continue                                # внутри сплошной речи границы не трогаем
        a = int((w['s'] - offset - 0.25) * 100); b = int((w['s'] - offset + 0.2) * 100)
        a, b = max(0, a), min(n, b)
        if b - a < 5: continue
        seg = env[a:b]; thr = 0.25 * env[a:min(n, b + 30)].max()
        over = np.nonzero(seg > thr)[0]
        if len(over):
            ns = (a + over[0]) / 100 + offset
            if i == 0 or ns > words[i - 1]['e']: w['s'] = round(min(ns, w['e'] - 0.05), 3)
    return words


def motion(video):
    """Насколько меняется картинка по секундам (0..1, ранг): меню, лобби и AFK — низкие значения.
    Декодируются только ключевые кадры, время берётся из showinfo — один быстрый проход."""
    p = subprocess.run(['ffmpeg', '-v', 'info', '-skip_frame', 'nokey', '-i', video, '-an', '-vf', 'scale=64:36,format=gray,showinfo',
                        '-fps_mode', 'vfr', '-f', 'rawvideo', '-'], capture_output=True)
    times = [float(t) for t in re.findall(rb'pts_time:(-?[0-9.]+)', p.stderr)]
    fr = np.frombuffer(p.stdout, np.uint8); k = min(len(fr) // (64 * 36), len(times))
    if k < 3: return None
    f = fr[:k * 64 * 36].reshape(k, 36, 64).astype(np.float32); t = np.maximum(0, np.array(times[:k]))
    d = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2)) / np.maximum(1.0, np.diff(t))
    sec = np.interp(np.arange(int(t[-1]) + 1), t[1:], d)
    return np.argsort(np.argsort(sec)) / max(1, len(sec) - 1)


def speech_regions(audio, threshold=0.35):
    """Где в куске звучит речь (silero VAD из faster-whisper): [(начало, конец)] в секундах от начала куска."""
    try:
        from faster_whisper.vad import get_speech_timestamps, VadOptions
        ts = get_speech_timestamps(audio, VadOptions(threshold=threshold, min_silence_duration_ms=250, speech_pad_ms=60))
        return [(t['start'] / SR, t['end'] / SR) for t in ts]
    except Exception:
        return []


def fill_and_tighten(words, seg, offset, lang, model, device, align_fn=None):
    """Доводка субтитров по VAD:
    1) куски речи ≥0.8 с, где нет ни одного слова (whisper пропустил на шуме), распознаём отдельно;
    2) начало слова после паузы, попавшее в тишину, переносим на начало речи (не дальше +0.5 с)."""
    regs = speech_regions(seg)
    if not regs: return words
    words = list(words)
    for a, b in regs:
        if b - a < 0.8: continue
        if any(a + offset - 0.1 <= w['s'] <= b + offset + 0.1 or w['s'] <= a + offset <= w['e'] for w in words): continue
        pa, pb = max(0.0, a - 0.3), min(len(seg) / SR, b + 0.3)
        piece = seg[int(pa * SR):int(pb * SR)]
        new = transcribe(piece, lang, model, offset=offset + pa, progress=False, device=device, vad_threshold=0.2)
        if new and align_fn:
            al = align_fn(new, piece, offset + pa, lang); new = al if al is not None else new
        words += [w for w in new if a + offset - 0.3 <= w['s'] <= b + offset + 0.3]
    words.sort(key=lambda w: w['s'])
    starts = [a + offset for a, b in regs]; spans = [(a + offset, b + offset) for a, b in regs]
    for i, w in enumerate(words):
        gap = w['s'] - words[i - 1]['e'] if i else 1.0
        if gap < 0.15 or any(x - 0.05 <= w['s'] <= y for x, y in spans): continue
        nxt = [t for t in starts if w['s'] < t <= w['s'] + 0.5]
        if nxt: w['s'] = round(min(nxt[0], w['e'] - 0.05), 3)
    return words
