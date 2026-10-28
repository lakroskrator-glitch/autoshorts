"""Субтитры ASS. Пресеты сняты с популярных нарезок со стримов и эдитов.

stream — жёлтый Montserrat, обычный регистр, тонкая обводка, по слову на стыке (самый частый формат нарезок)
soft   — белый скруглённый Nunito строчными с мягкой тенью
italic — белый курсив с тёмным свечением
bold / tilt / clean — крупный капс, акценты жёлтым
glow   — слово с неоновым свечением (эдиты)
cinema — фразы тонким разреженным капсом, акцент неоном (кино-эдиты)
События без речи (*смеётся*, *кричит*) — красным курсивом капсом.
"""
import random, re

WHITE = '&H00FFFFFF&'; BLACK = '&H00000000&'; YELLOW = '&H0000E1FF&'
NEON = ['&H00FF40C8&', '&H00FFB040&', '&H004060FF&']

# font, size(для ширины 1080), цвет, обводка, цвет обводки, тень, размытие, регистр, режим, якорь, акцент
STYLES = {
    'stream': dict(font='Montserrat ExtraBold', size=112, col=YELLOW, bord=4, bcol=BLACK, shad=2, blur=0.6, case='lower', mode='word', an=5, accent=None),
    'soft':   dict(font='Nunito Black', size=118, col=WHITE, bord=0, bcol=BLACK, shad=4, blur=2.5, case='lower', mode='word', an=2, accent=None),
    'italic': dict(font='Montserrat ExtraBold', italic=True, size=110, col=WHITE, bord=5, bcol='&H40000000&', shad=0, blur=5, case='lower', mode='word', an=5, accent=None),
    'bold':   dict(font='Montserrat Black', size=128, col=WHITE, bord=7, bcol=BLACK, shad=3, blur=0.6, case='upper', mode='word', an=5, accent=YELLOW),
    'tilt':   dict(font='Montserrat Black', size=124, col=WHITE, bord=7, bcol=BLACK, shad=3, blur=0.6, case='upper', mode='word', an=5, accent=YELLOW, tilt=4),
    'clean':  dict(font='Montserrat ExtraBold', size=112, col=WHITE, bord=5, bcol=BLACK, shad=2, blur=0.6, case='upper', mode='word', an=5, accent=YELLOW),
    'glow':   dict(font='Montserrat Black', size=124, col=WHITE, bord=0, bcol=BLACK, shad=0, blur=0, case='upper', mode='glow', an=5, accent=None),
    'cinema': dict(font='Montserrat ExtraBold', size=58, col=WHITE, bord=2, bcol=BLACK, shad=3, blur=0.6, case='upper', mode='phrase', an=5, accent=None),
}
EVENT = dict(font='Montserrat Black', size=96, col='&H002030FF&', bord=5, bcol='&H00000050&', shad=3)


def _t(t):
    t = max(0.0, t); h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f'{h}:{m:02d}:{s:05.2f}'


def _esc(s):
    return re.sub(r'[{}\\]', '', s)


def _core(w):
    return re.sub(r'[^\w-]', '', w.lower())


def is_emph(w, emph):
    core = _core(w['w'])
    return core in emph or w.get('loud') or bool(re.search(r'\d', core)) or w['w'].endswith('!')


def _clean(w, case):
    t = _esc(w['w'].strip('«»"„“')).strip('.,;:…!?-—')
    return t.upper() if case == 'upper' else t.lower() if case == 'lower' else t


def chunks(words, max_words=4, max_chars=22):
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(w); nxt = words[i + 1] if i + 1 < len(words) else None
        if (len(cur) >= max_words or len(' '.join(x['w'] for x in cur)) > max_chars or nxt is None
                or re.search(r'[.!?…,]$', w['w']) or nxt['s'] - w['e'] > 0.5):
            out.append(cur); cur = []
    return out


def build(words, t0, path, y, style='stream', title=None, emphasis=(), max_gap=1.2, W=1080, H=1920, events=()):
    st = STYLES.get(style, STYLES['stream']); k = min(W, H) / 1080
    size = int(st['size'] * k); bord = st['bord'] * k; shad = st['shad'] * k
    emph = {_core(e) for e in emphasis}; rnd = random.Random(7); ev = []
    mode = st['mode']; spacing = 7 if mode == 'phrase' else 0.5
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Word,{st['font']},{size},{st['col']},{st['col']},{st['bcol']},&H70000000&,0,{-1 if st.get('italic') else 0},0,0,100,100,{spacing},0,1,{bord:.1f},{shad:.1f},5,40,40,0,1
Style: Event,{EVENT['font']},{int(EVENT['size'] * k)},{EVENT['col']},{EVENT['col']},{EVENT['bcol']},&H90000000&,0,-1,0,0,100,100,1,0,1,{EVENT['bord'] * k:.1f},{EVENT['shad'] * k:.1f},5,40,40,0,1
Style: TitleC,Montserrat ExtraBold,{int(54 * k)},&H00FFFFFF&,&H00FFFFFF&,&H00FF40C8&,&H66000000&,0,0,0,0,100,100,8,0,1,0,2,8,60,60,{int(H * 0.08)},1
Style: Title,Montserrat Black,{int(60 * k)},&H00101010&,&H00101010&,&H0000E1FF&,&H66000000&,0,0,0,0,100,100,0,0,3,22,0,8,60,60,{int(H * 0.057)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    an = st['an']; yy = y + int(8 * k) if an == 2 else y
    pos = fr'\pos({W // 2},{yy})\an{an}\blur{st["blur"]}'

    def end_of(i, seq, s):
        nxt = seq[i + 1]['s'] - t0 if i + 1 < len(seq) else seq[i]['e'] - t0 + 0.4
        e = min(nxt, s + max_gap, seq[i]['e'] - t0 + 0.6) if nxt > s else s + 0.25
        return max(e, s + 0.12)

    words = [dict(w) for w in words]                              # слова с одинаковым началом не рисуем друг на друге
    for i in range(1, len(words)):
        if words[i]['s'] < words[i - 1]['s'] + 0.1: words[i]['s'] = round(words[i - 1]['s'] + 0.1, 3)
    if mode in ('word', 'glow'):
        for i, w in enumerate(words):
            s = w['s'] - t0; e = end_of(i, words, s); txt = _clean(w, st['case'])
            n = len(txt); lim = 11 if st['case'] != 'upper' else 9
            fs = size if n <= lim else int(size * lim / n) + int(6 * k)
            hot = is_emph(w, emph)
            rot = rnd.uniform(-st.get('tilt', 0), st.get('tilt', 0))
            grow = 112 if hot else 104                       # громкие/акцентные слова «прыгают» сильнее
            pop = fr'\fscx88\fscy88\t(0,60,\fscx{grow}\fscy{grow})\t(60,140,\fscx100\fscy100)'
            if mode == 'word':
                col = fr'\c{st["accent"]}' if hot and st['accent'] else ''
                ev.append(f'Dialogue: 0,{_t(s)},{_t(e)},Word,,0,0,0,,{{{pos}\\fs{fs}{col}\\frz{rot:.1f}{pop}}}{txt}')
            else:
                c = NEON[i % len(NEON)] if hot else NEON[0]
                ev.append(f'Dialogue: 0,{_t(s)},{_t(e)},Word,,0,0,0,,{{{pos}\\fs{fs}\\c{c}\\3c{c}\\bord{int(10 * k)}\\blur{int(14 * k)}\\alpha&H30&{pop}}}{txt}')
                ev.append(f'Dialogue: 1,{_t(s)},{_t(e)},Word,,0,0,0,,{{{pos}\\fs{fs}\\c{WHITE}\\bord{max(1, int(2 * k))}\\3c{c}{pop}}}{txt}')
    else:
        chs = chunks(words)
        for ci, ch in enumerate(chs):
            s = ch[0]['s'] - t0; e = min(ch[-1]['e'] - t0 + 0.35, s + 4.0)
            if ci + 1 < len(chs): e = min(e, chs[ci + 1][0]['s'] - t0)
            parts = []
            for w in ch:
                t = _clean(w, st['case'])
                if is_emph(w, emph):
                    t = fr'{{\c&H00FFFFFF&\3c{NEON[0]}\bord{int(6 * k)}\blur{int(8 * k)}}}' + t + r'{\bord0\blur0.6}'
                parts.append(t)
            ev.append(f'Dialogue: 0,{_t(s)},{_t(e)},Word,,0,0,0,,{{{pos}\\4a&H60&\\fad(120,120)\\fscx96\\fscy96\\t(0,200,\\fscx100\\fscy100)}}' + ' '.join(parts))
    for s, e, text in events:                                     # *смеётся*, *кричит* — там, где нет слов
        ev.append(f'Dialogue: 2,{_t(s - t0)},{_t(e - t0)},Event,,0,0,0,,{{\\pos({W // 2},{y})\\an5\\fscx80\\fscy80\\t(0,90,\\fscx108\\fscy108)\\t(90,170,\\fscx100\\fscy100)}}*{_esc(text).upper()}*')
    if title:
        if mode == 'phrase':
            ev.append(f'Dialogue: 3,{_t(0.2)},{_t(3.4)},TitleC,,0,0,0,,' + r'{\fad(300,300)\blur0.8}' + _esc(title).upper())
        else:
            ev.append(f'Dialogue: 3,{_t(0)},{_t(3.2)},Title,,0,0,0,,' + r'{\fad(150,250)\t(0,160,\fscx104\fscy104)\t(160,300,\fscx100\fscy100)}' + _esc(title).upper())
    open(path, 'w', encoding='utf-8').write(head + '\n'.join(ev) + '\n')
