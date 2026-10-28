"""Поиск лучших моментов: эвристика (крючки, крики, смех, мат, темп речи) или Claude (по смыслу, с темой ролика)."""
import json, os, re
import numpy as np

HOOKS = {
    'ru': ['как', 'почему', 'зачем', 'секрет', 'ошибк', 'никогда', 'всегда', 'главн', 'лучш', 'худш', 'деньг', 'бесплатн',
           'правд', 'миф', 'серьёзно', 'реально', 'представ', 'короче', 'смотри', 'внимание', 'удивл', 'проблем', 'тысяч', 'миллион'],
    'en': ['how', 'why', 'secret', 'mistake', 'never', 'always', 'best', 'worst', 'money', 'free', 'truth', 'seriously',
           'imagine', 'look', 'insane', 'crazy', 'problem', 'thousand', 'million'],
}
HYPE = {
    'ru': ['ахах', 'хаха', 'аха', 'ору', 'лол', 'бля', 'пизд', 'нихуя', 'охуе', 'ебать', 'ёб', 'еба', 'жесть', 'капец', 'офиге',
           'что?', 'да ладно', 'не может быть', 'вау', 'ого', 'чего'],
    'en': ['haha', 'lol', 'lmao', 'wtf', 'omg', 'no way', 'what?', 'holy', 'damn', 'shit', 'fuck', 'bro', 'wow'],
}


FUNNY = {
    'ru': ['ахах', 'хаха', 'ха-ха', 'хах', 'пхах', 'хех', 'ору', 'ржу', 'угар', 'смешно', 'рофл', 'кринж', 'лол', 'аха'],
    'en': ['haha', 'lol', 'lmao', 'rofl', 'funny', 'hilarious', 'dying', 'bruh'],
}


def laugh_curve(audio, sr=16000, step=0.5):
    """Похожесть на смех по окнам step: ритмичные всплески громкости 3–8 Гц (ха-ха-ха) при заметной громкости."""
    hop = sr // 100; n = len(audio) // hop                        # огибающая 100 Гц, блоками — без копии всего аудио
    if n < 200: return np.zeros(max(1, int(len(audio) / sr / step)))
    env = np.empty(n, np.float32)
    for i in range(0, n, 100000):
        j = min(n, i + 100000); env[i:j] = np.abs(audio[i * hop:j * hop]).reshape(j - i, hop).mean(axis=1)
    win = 200; out = []                                            # окно 2 с
    freqs = np.fft.rfftfreq(win, 0.01); band = (freqs >= 3) & (freqs <= 8); allb = (freqs >= 0.5) & (freqs <= 20)
    hann = np.hanning(win)
    for c in np.arange(0, n / 100, step):
        a = int(c * 100) - win // 2; seg = env[max(0, a):max(0, a) + win]
        if len(seg) < win: seg = np.pad(seg, (0, win - len(seg)))
        seg = (seg - seg.mean()) * hann; sp = np.abs(np.fft.rfft(seg)) ** 2
        out.append(sp[band].sum() / (sp[allb].sum() + 1e-12) * np.sqrt(env[max(0, a):max(0, a) + win].mean() + 1e-9))
    x = np.array(out); return (x - np.median(x)) / (np.std(x) + 1e-9)


def heuristic_funny(words, loud, laugh, step, lang, n, dmin=5, dmax=20, motion=None):
    """Смешные моменты: шутка + смех после неё (смех в звуке и «ахах» в речи)."""
    hype = hype_curve(loud, step); sents = sentences(words); cands = []
    for i in range(len(sents)):
        ws = []
        for j in range(i, len(sents)):
            ws = ws + sents[j]; d = ws[-1]['e'] - ws[0]['s']
            if d > dmax: break
            if d < dmin: continue
            text = ' '.join(w['w'] for w in ws).lower()
            a, b = int(ws[0]['s'] / step), int((ws[-1]['e'] + 2.0) / step) + 1      # смех часто сразу после панчлайна
            lg = laugh[a:b]; hp = hype[a:b]
            sc = (3.0 * _hits(text, FUNNY.get(lang, FUNNY['en'])) + 1.2 * _hits(text, HYPE.get(lang, HYPE['en']))
                  + 1.5 * float(np.sum(np.clip(lg, 1.5, None) - 1.5)) + 0.8 * float(np.sum(np.clip(hp, 1.5, None) - 1.5))
                  + 0.5 * (text.count('!') + text.count('?')) - 0.05 * d) * motion_k(ws, motion, 1.0)
            cands.append((sc, ws))
    cands.sort(key=lambda c: -c[0]); chosen = []
    for sc, ws in cands:
        s, e = ws[0]['s'], ws[-1]['e'] + 1.2
        if sc <= 0: break
        if all(e + 2 <= c['start'] or s >= c['end'] + 2 for c in chosen):
            chosen.append({'start': s, 'end': e, 'score': round(sc, 2), 'title': short_title(ws, 32), 'hook': '', 'description': '',
                           'hashtags': [], 'emphasis': [], 'why': 'эвристика смеха'})
        if len(chosen) == n: break
    return sorted(chosen, key=lambda c: c['start'])


def sentences(words, pause=0.7, max_len=15.0):
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(w); nxt = words[i + 1] if i + 1 < len(words) else None
        long_ = cur[-1]['e'] - cur[0]['s'] > max_len and (nxt is None or nxt['s'] - w['e'] > 0.2 or w['w'].endswith(','))
        if re.search(r'[.!?…]$', w['w']) or nxt is None or nxt['s'] - w['e'] > pause or long_:
            out.append(cur); cur = []
    return out


WHOLE = {'ого', 'ору', 'ржу', 'бля', 'лол', 'аха', 'хах', 'хех', 'еба', 'ёб', 'вау', 'как', 'миф', 'топ', 'bro', 'wow', 'omg', 'lol', 'how', 'why', 'look', 'free', 'best'}
_rx = {}


def _hits(text, vocab):
    """Совпадения по началу слова (ошибк → ошибка, ошибки), короткие междометия — только целым словом («ого» ≠ «много»)."""
    n = 0
    for h in vocab:
        if h not in _rx:
            core = re.escape(h.rstrip('?'))
            _rx[h] = re.compile(r'(?<!\w)' + core + (r'(?!\w)' if h in WHOLE else '') + (r'\?' if h.endswith('?') else ''))
        n += len(_rx[h].findall(text))
    return n


def hype_curve(loud, step=0.5):
    """z-оценка громкости относительно скользящего фона ±60 с: всплески = крики/смех."""
    k = max(1, int(60 / step)); pad = np.pad(loud, k, mode='edge')
    base = np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode='same')[k:-k]
    sd = np.std(loud) + 1e-6
    return (loud - base) / sd


def motion_k(ws, motion, extra=0.0):
    """Множитель оценки 0.2..1: статичная картинка (меню, лобби, AFK) сильно топит момент."""
    if motion is None or not len(motion): return 1.0
    m = motion[int(ws[0]['s']):int(ws[-1]['e'] + extra) + 1]
    return float(np.clip((m.mean() / 0.55) ** 2, 0.2, 1.0)) if len(m) else 1.0


def motion_pen(ws, motion, extra=0.0):
    """Штраф за статичную картинку (меню, лобби, AFK): motion — средняя разница кадров по секундам, 0..1 ранг."""
    if motion is None or not len(motion): return 0.0
    a, b = int(ws[0]['s']), int(ws[-1]['e'] + extra) + 1
    m = motion[a:b]
    return float(np.clip(0.55 - m.mean(), 0, None) * 30) if len(m) else 0.0


def score_window(ws, hype, step, lang, motion=None):
    text = ' '.join(w['w'] for w in ws).lower(); dur = ws[-1]['e'] - ws[0]['s']
    first = ' '.join(w['w'] for w in ws[:12]).lower()
    a, b = int(ws[0]['s'] / step), max(int(ws[0]['s'] / step) + 1, int(ws[-1]['e'] / step))
    h = hype[a:b] if b <= len(hype) else hype[a:]
    peaks = float(np.sum(np.clip(h, 1.5, None) - 1.5)) if len(h) else 0
    gaps = sum(max(0, y['s'] - x['e'] - 0.6) for x, y in zip(ws, ws[1:]))
    rate = len(ws) / max(dur, 1e-3)
    return (1.0 * _hits(text, HOOKS.get(lang, HOOKS['en'])) + 2.0 * _hits(first, HOOKS.get(lang, HOOKS['en']))
            + 1.6 * _hits(text, HYPE.get(lang, HYPE['en'])) + 0.8 * (text.count('!') + text.count('?'))
            + 1.2 * rate + 1.5 * peaks - 1.2 * gaps) * motion_k(ws, motion)


def heuristic(words, loud, step, lang, n, dmin, dmax, motion=None):
    hype = hype_curve(loud, step); sents = sentences(words); cands = []
    for i in range(len(sents)):
        ws = []
        for j in range(i, len(sents)):
            ws = ws + sents[j]; d = ws[-1]['e'] - ws[0]['s']
            if d > dmax: break
            if d >= dmin: cands.append((score_window(ws, hype, step, lang, motion), ws))
    cands.sort(key=lambda c: -c[0]); chosen = []
    for sc, ws in cands:
        s, e = ws[0]['s'], ws[-1]['e']
        if all(e + 5 <= c['start'] or s >= c['end'] + 5 for c in chosen):
            voc = HOOKS.get(lang, HOOKS['en']) + HYPE.get(lang, HYPE['en'])
            first = max(sentences(ws), key=lambda x: (len({w['w'].lower() for w in x}) >= 3, _hits(' '.join(w['w'] for w in x).lower(), voc), len(x)))
            chosen.append({'start': s, 'end': e, 'score': round(sc, 2), 'title': short_title(first), 'hook': '', 'description': '',
                           'hashtags': [], 'emphasis': [], 'why': 'эвристика'})
        if len(chosen) == n: break
    if not chosen and words and words[-1]['e'] - words[0]['s'] < dmin:   # речь короче минимума — берём её целиком
        chosen = [{'start': words[0]['s'], 'end': min(words[-1]['e'], words[0]['s'] + dmax), 'score': 0, 'title': short_title(words[:8]), 'hook': '',
                   'description': '', 'hashtags': [], 'emphasis': [], 'why': 'видео короче минимума'}]
    return sorted(chosen, key=lambda c: c['start'])


def short_title(ws, limit=42):
    """Заголовок из слов фразы: без подряд идущих повторов («реально, реально»), не длиннее limit, без обрыва слова."""
    uniq, prev = [], None
    for w in ws:
        core = re.sub(r'\W', '', w['w'].lower())
        if core and core == prev: continue
        uniq.append(w); prev = core
    t = ''
    for w in uniq:
        if len(t) + len(w['w']) > limit: return t.rstrip(' ,') + '…'
        t += w['w'] + ' '
    return t.strip().rstrip(',')


# ---------------- Claude ----------------
SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['compilation_title', 'clips'],
    'properties': {'compilation_title': {'type': 'string', 'description': 'название всей сборки (для режима сборки), до 50 символов'},
                   'clips': {'type': 'array', 'items': {
        'type': 'object', 'additionalProperties': False,
        'required': ['start', 'end', 'title', 'hook', 'description', 'hashtags', 'emphasis', 'why', 'score'],
        'properties': {
            'start': {'type': 'number', 'description': 'секунда начала (из меток транскрипта)'},
            'end': {'type': 'number', 'description': 'секунда конца'},
            'title': {'type': 'string', 'description': 'тема ролика: короткий цепляющий заголовок, до 45 символов'},
            'hook': {'type': 'string', 'description': 'надпись-крючок на первые 3 секунды, до 35 символов'},
            'description': {'type': 'string', 'description': 'подпись к публикации, 1-2 предложения'},
            'hashtags': {'type': 'array', 'items': {'type': 'string'}},
            'emphasis': {'type': 'array', 'items': {'type': 'string'}, 'description': 'слова из речи для жёлтого выделения'},
            'why': {'type': 'string', 'description': 'почему момент сработает'},
            'score': {'type': 'integer', 'description': 'оценка вирусности 1-10'},
        }}}},
}

SYSTEM = """Ты — монтажёр вирусных нарезок со стримов для TikTok, Reels и YouTube Shorts.
Тебе дают расшифровку стрима с метками времени в секундах. Метка 🔊 значит всплеск громкости (крик, смех, эмоция).
Выбери самые сильные самостоятельные моменты: смешные, эмоциональные, скандальные, неожиданные, с историей или панчлайном.
Требования к моменту:
- понятен без контекста стрима; начинается с сильной фразы (крючок в первые 3 секунды), а не с середины мысли;
- заканчивается на панчлайне/реакции, не обрывая фразу;
- длина в заданных пределах; моменты не пересекаются.
Заголовок (title) — тема ролика в стиле TikTok, на языке стрима, без кликбейта-лжи. Хэштеги без символа #."""


FUNNY_SYSTEM = """Ты — монтажёр сборок «смешные моменты со стрима» для YouTube и TikTok.
Тебе дают расшифровку стрима с метками времени в секундах. 🔊 — всплеск громкости, 😂 — похоже на смех.
Выбери самые смешные короткие моменты: шутка, нелепость, реакция, панчлайн — и обязательно смех/реакция после него.
Каждый момент: понятен без контекста, начинается чуть раньше шутки (с подводки), заканчивается после реакции.
title — короткая подпись к моменту (до 32 символов) для экрана; compilation_title — название всей сборки.
Хэштеги без символа #."""


def transcript_lines(words, loud, step, span=8.0, laugh=None):
    hype = hype_curve(loud, step); lines, cur, t0 = [], [], None
    for w in words:
        if t0 is None: t0 = w['s']
        cur.append(w['w'])
        if w['e'] - t0 >= span:
            a, b = int(t0 / step), int(w['e'] / step) + 1
            mark = ' 🔊' if len(hype[a:b]) and hype[a:b].max() > 2.0 else ''
            if laugh is not None and len(laugh[a:b]) and laugh[a:b].max() > 2.0: mark += ' 😂'
            lines.append(f'[{t0:.1f}] {" ".join(cur)}{mark}'); cur, t0 = [], None
    if cur: lines.append(f'[{t0:.1f}] {" ".join(cur)}')
    return '\n'.join(lines)


def with_claude(words, loud, step, lang, n, dmin, dmax, model='claude-opus-5-5', topic=None, funny=False, laugh=None):
    try:
        import anthropic
    except ImportError:
        raise SystemExit('Для --ai нужен пакет anthropic: pip install anthropic')
    user = (f'Язык стрима: {lang}. Нужно {n} моментов длиной от {dmin:.0f} до {dmax:.0f} секунд.'
            + (f' Тематика канала: {topic}.' if topic else '') + '\n\n<transcript>\n' + transcript_lines(words, loud, step, laugh=laugh) + '\n</transcript>')
    try:
        client = anthropic.Anthropic()
        r = client.beta.messages.create(
            model=model, max_tokens=16000, system=FUNNY_SYSTEM if funny else SYSTEM,
            messages=[{'role': 'user', 'content': user}],
            thinking={'type': 'adaptive'},
            output_config={'effort': 'medium', 'format': {'type': 'json_schema', 'schema': SCHEMA}},
            betas=['server-side-fallback-2026-07-01'], fallbacks='default',
        )
    except anthropic.AuthenticationError:
        raise SystemExit('Claude: неверный ключ. Задайте ANTHROPIC_API_KEY или запустите без --ai.')
    except anthropic.RateLimitError:
        raise SystemExit('Claude: превышен лимит запросов, попробуйте позже.')
    except anthropic.APIStatusError as e:
        raise SystemExit(f'Claude: ошибка API {e.status_code}: {e.message}')
    except anthropic.APITimeoutError:
        raise SystemExit('Claude: превышено время ожидания ответа, попробуйте ещё раз.')
    except anthropic.APIConnectionError:
        raise SystemExit('Claude: нет соединения с API.')
    if r.stop_reason == 'refusal':
        raise SystemExit('Claude отказался обрабатывать этот стрим (refusal). Запустите без --ai.')
    if r.stop_reason == 'max_tokens':
        raise SystemExit('Claude: ответ обрезан (max_tokens).')
    text = next((b.text for b in r.content if b.type == 'text'), None)
    if not text: raise SystemExit('Claude: пустой ответ, запустите ещё раз или без --ai.')
    data = json.loads(text); clips = []
    for c in sorted(data['clips'], key=lambda c: -c.get('score', 0)):          # лучшие первыми
        ws = [w for w in words if w['s'] >= c['start'] - 0.5 and w['e'] <= c['end'] + 0.8]
        if len(ws) < 3: continue
        c['start'], c['end'] = ws[0]['s'], ws[-1]['e']
        if c['end'] - c['start'] > dmax * 1.15:                                     # длиннее разрешённого — обрезаем по слову
            ws = [w for w in ws if w['e'] <= c['start'] + dmax]; c['end'] = ws[-1]['e']
        if c['end'] - c['start'] < dmin * 0.6: continue
        if all(c['end'] <= o['start'] or c['start'] >= o['end'] for o in clips): clips.append(c)
        if len(clips) == n: break
    clips = sorted(clips, key=lambda c: c['start'])
    for c in clips: c['compilation_title'] = data.get('compilation_title', '')
    return clips


def ai_available():
    """None — всё есть; иначе текст, чего не хватает."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return 'Для --ai нужен пакет anthropic: pip install anthropic'
    return None if bool(os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('ANTHROPIC_AUTH_TOKEN') or os.environ.get('ANTHROPIC_PROFILE')
                or os.path.isdir(os.path.expanduser('~/.config/anthropic'))) else 'Для --ai нужен ключ Claude API: переменная ANTHROPIC_API_KEY (или ant auth login)'


def trim_static_start(clips, words, motion, dmin):
    """Если момент начинается с меню/лобби (статичная картинка), а дальше идёт действие — сдвигаем начало
    на первую фразу после статичного куска, пока длина остаётся ≥ dmin."""
    if motion is None or not len(motion): return clips
    for c in clips:
        a = int(c['start']); m = motion[a:a + 10]
        if len(m) < 4 or m[:3].mean() > 0.35: continue
        moving = [i for i in range(len(m)) if m[i:i + 2].mean() > 0.45]
        if not moving: continue
        t = a + moving[0]
        talk = sum(1 for w in words if c['start'] <= w['s'] < t)
        if talk > 1.2 * max(1.0, t - c['start']): continue        # в «статичном» куске говорят — это студия/общение, а не меню
        starts = [w['s'] for i, w in enumerate(words) if c['start'] < w['s'] <= t + 1.5 and (i == 0 or w['s'] - words[i - 1]['e'] > 0.3)]
        ns = max([x for x in starts if x <= t + 1.5] or [t])
        if c['end'] - ns >= dmin: c['start'] = round(ns, 3); c['why'] = c.get('why', '') + '; начало сдвинуто с меню'
    return clips


def fix_bounds(clips, words, dmax, pause=0.35, back=6.0, fwd=4.0, laugh=None, hype=None, step=0.5, payoff=8.0):
    """Контролёр границ: момент не должен начинаться и обрываться посреди фразы.
    Начало сдвигается назад к ближайшей паузе (до back с), конец — вперёд к паузе (до fwd с), пока длина ≤ dmax·1.1."""
    if not words: return clips
    for c in clips:
        idx = [i for i, w in enumerate(words) if c['start'] - 0.05 <= w['s'] <= c['end']]
        if not idx: continue
        i, j = idx[0], idx[-1]; s0, e0 = c['start'], c['end']; fixed = []
        k = i
        while k > 0 and words[k]['s'] - words[k - 1]['e'] < pause and s0 - words[k - 1]['s'] <= back \
                and c['end'] - words[k - 1]['s'] <= dmax * 1.1: k -= 1
        if k < i and (k == 0 or words[k]['s'] - words[k - 1]['e'] >= pause):
            c['start'] = words[k]['s']; fixed.append(f'начало −{s0 - c["start"]:.1f} с')
        k = j
        while k + 1 < len(words) and words[k + 1]['s'] - words[k]['e'] < pause and words[k + 1]['e'] - e0 <= fwd \
                and words[k + 1]['e'] - c['start'] <= dmax * 1.1: k += 1
        if k > j and (k + 1 == len(words) or words[k + 1]['s'] - words[k]['e'] >= pause):
            c['end'] = max(c['end'], words[k]['e'] + 0.4); fixed.append(f'конец +{c["end"] - e0:.1f} с')
        if laugh is not None or hype is not None:            # развязка сразу после конца (смех, крик) — дотягиваем до неё
            a, b = int(c['end'] / step), int((c['end'] + payoff) / step)
            peaks = [k for k in range(a, b) if (laugh is not None and k < len(laugh) and laugh[k] > 2.2)
                     or (hype is not None and k < len(hype) and hype[k] > 2.5)]
            if peaks:
                t = peaks[0] * step + 1.0; e1 = c['end']
                nxt = [w for w in words if w['s'] > c['end'] and w['s'] <= t + 3]
                for k, w in enumerate(nxt):                    # конец — на паузе после реакции
                    if w['e'] >= t and (k + 1 == len(nxt) or nxt[k + 1]['s'] - w['e'] >= pause): e1 = w['e'] + 0.4; break
                else: e1 = max(e1, t)
                if e1 > c['end'] and e1 - c['start'] <= dmax * 1.1:
                    fixed.append(f'развязка +{e1 - c["end"]:.1f} с'); c['end'] = e1
        if fixed: c['why'] = (c.get('why') or '') + '; контролёр: фраза целиком (' + ', '.join(fixed) + ')'
    return clips

