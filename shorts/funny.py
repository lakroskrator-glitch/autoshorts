"""Отбор по-настоящему смешных моментов (без интернета и ключей).

1. Кандидаты — эвристика смеха (реакции в речи, ритм «ха-ха» в звуке, крики) с запасом: в 3 раза больше нужного.
2. Каждый кандидат точно распознаётся заново (whisper turbo), чтобы был понятен смысл шутки.
3. Нейросеть AudioSet (AST) слушает момент и несколько секунд после него: смех, хихиканье, крики.
4. Локальная языковая модель (Qwen3-4B, llama.cpp) в роли трёх судей — «ироник», «панчлайн», «зритель TikTok» —
   ранжирует кандидатов (каждый в своём случайном порядке, чтобы не влиял порядок), «редактор» выбирает итог из финалистов
   и пишет ироничные подписи.
Итог — сумма трёх оценок. Без llama-cpp-python или torch соответствующий шаг пропускается.
"""
import json, os, random, re, sys
import numpy as np

AST = 'MIT/ast-finetuned-audioset-10-10-0.4593'
LAUGH = [16, 17, 18, 19, 20, 21]          # Laughter, Baby laughter, Giggle, Snicker, Belly laugh, Chuckle
SCREAM = [8, 11, 14]                      # Shout, Yell, Screaming
SING = [27, 32, 33, 34]                # Singing, Male/Female/Child singing — песни не смешные
LLM_REPO, LLM_FILE = 'unsloth/Qwen3-4B-Instruct-2507-GGUF', 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
_ast = None


def _log(msg): print(msg, file=sys.stderr, flush=True)


def audio_scores(cands, load):
    """AST по окнам 10 с на [начало, конец + 5 с]: максимум смеха, крика и музыки. load(s, d) → аудио 16 кГц."""
    global _ast
    try:
        import torch, warnings
        warnings.filterwarnings('ignore', message='.*mel filter has all zero values.*')
        from transformers import ASTFeatureExtractor, ASTForAudioClassification, logging as tlog
        tlog.set_verbosity_error(); tlog.disable_progress_bar()
        if _ast is None:
            _ast = (ASTFeatureExtractor.from_pretrained(AST), ASTForAudioClassification.from_pretrained(AST).eval())
    except Exception as e:
        _log(f'  (детектор смеха AST недоступен: {type(e).__name__}; оцениваю без него)'); return False
    fe, m = _ast
    for c in cands:
        a = load(c['start'], c['end'] - c['start'] + 5.0)
        segs = [a[i:i + 160000] for i in range(0, max(1, len(a) - 60000), 80000)]
        with torch.inference_mode():
            p = torch.sigmoid(m(**fe(segs, sampling_rate=16000, return_tensors='pt')).logits).numpy()
        c['laugh_p'] = round(float(p[:, LAUGH].max()), 3); c['scream_p'] = round(float(p[:, SCREAM].max()), 3)
        c['sing_p'] = round(float(p[:, SING].max(axis=1).mean()), 3)
    return True


def _ast_windows(starts, load, d=10.0):
    """Вероятность смеха в окнах [t, t + d] — для поиска пропущенного смеха по пикам laugh_curve."""
    cs = [{'start': t, 'end': t + d - 5.0} for t in starts]
    return [c['laugh_p'] for c in cs] if audio_scores(cs, load) else None


def laugh_candidates(words, laugh, step, load, dmin, dmax, sentences, k=40, taken=()):
    """Смех, который не нашла эвристика: берём k пиков ритма «ха-ха» в звуке, проверяем их AST и для
    подтверждённых строим момент из фраз, которые заканчиваются прямо перед смехом."""
    order = np.argsort(-laugh); peaks = []
    for i in order:
        t = i * step
        if laugh[i] < 1.5 or len(peaks) >= k: break
        if all(abs(t - p) > 15 for p in peaks) and all(not (c['start'] - 3 <= t <= c['end'] + 6) for c in taken): peaks.append(t)
    if not peaks: return []
    probs = _ast_windows([max(0.0, t - 5.0) for t in peaks], load)
    if not probs: return []
    sents = sentences(words); out = []
    for t, pr in zip(peaks, probs):
        if pr < 0.08: continue
        end = [j for j, s in enumerate(sents) if s[-1]['e'] <= t + 1.5]
        if not end or t - sents[end[-1]][-1]['e'] > 5: continue
        j = end[-1]; i = j
        while i > 0 and sents[j][-1]['e'] - sents[i][0]['s'] < dmin and sents[j][-1]['e'] - sents[i - 1][0]['s'] <= dmax: i -= 1
        ws = [w for s in sents[i:j + 1] for w in s]
        if ws[-1]['e'] - ws[0]['s'] < dmin * 0.6: continue
        out.append({'start': ws[0]['s'], 'end': max(ws[-1]['e'] + 1.2, min(t + 1.5, ws[-1]['e'] + 3)), 'score': 0.0,
                    'text': ' '.join(w['w'] for w in ws), 'title': '', 'hook': '', 'description': '', 'hashtags': [], 'emphasis': []})
    return out


def llm_path(download=True):
    """Путь к GGUF: AUTOSHORTS_LLM, папка models/ рядом с программой или кэш Hugging Face (скачает ~2.5 ГБ один раз)."""
    env = os.environ.get('AUTOSHORTS_LLM')
    if env: return env if os.path.exists(env) else None
    local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', LLM_FILE)
    if os.path.exists(local): return local
    try:
        from huggingface_hub import hf_hub_download
        try:
            return hf_hub_download(LLM_REPO, LLM_FILE, local_files_only=True)
        except Exception:
            if not download: return None
            _log(f'  скачиваю языковую модель Qwen3-4B (~2.5 ГБ, один раз)…')
            return hf_hub_download(LLM_REPO, LLM_FILE)
    except Exception as e:
        _log(f'  (языковая модель недоступна: {type(e).__name__}: {str(e)[:100]})'); return None


# «Агенты» — судьи с разным взглядом на смешное (критерии сняты с референсов: вирусные нарезки стримов,
# где смешно от иронии, подколов и реакции, а не от громкости). Каждый ранжирует всех кандидатов списком,
# голоса складываются (метод Борда), затем «редактор» смотрит финалистов, выбирает итог и пишет подписи.
JUDGES = {
    'ru': {
        'ироник': """Ты — судья по ИРОНИИ. Ищи, где слова расходятся с происходящим: «ну всё, я профи» — и тут же провал;
сарказм («отличный план, конечно»); самоирония; серьёзный тон про нелепую вещь; обещание, которое тут же нарушается.""",
        'панчлайн': """Ты — судья по ПАНЧЛАЙНУ. Ищи структуру «подводка → неожиданный поворот → реакция»: история с концовкой,
подкол друга и его ответ, абсурдный вопрос или комментарий, после которого смеются.""",
        'зритель': """Ты — обычный зритель TikTok, листающий ленту. Выбирай то, над чем ты засмеёшься БЕЗ контекста стрима
за первые секунды: бурная эмоция, нелепый провал, смешная фраза, которую хочется переслать другу.""",
    },
    'en': {
        'irony': """You judge IRONY: words contradicting reality ("I'm a pro" then an instant fail), sarcasm,
self-irony, a serious tone about something silly, a promise broken right away.""",
        'punchline': """You judge PUNCHLINES: setup → unexpected twist → reaction; a story with a payoff,
roasting a friend and the comeback, an absurd question or comment that gets a laugh.""",
        'viewer': """You are a regular TikTok viewer scrolling. Pick what makes you laugh WITHOUT stream context
in the first seconds: a wild emotion, an absurd fail, a line you'd send to a friend.""",
    },
}
COMMON = {
    'ru': """Ниже фрагменты стрима (расшифровка автоматическая, с ошибками — додумывай смысл). Пометки: [смех] — в звуке
слышен смех, [крик] — громкая реакция. Не смешно: песни и музыка, тактические переговоры в игре, бессвязный набор слов,
приветствия и прощания, обрывки фраз без смысла.
Пример смешного: «Я просто постеснялся в 2020 танцевать в тиктоке. Поэтому сейчас я с таким лицом еду на работу» — самоирония.
Пример смешного: «Всё, сейчас я их всех... ой. Я умер» — обещание и мгновенный провал.
Пример НЕсмешного: «Двое на Б, у него AWP, я за ящиком» — просто игра. «Видела, видела» — нет шутки.
Выбери {k} самых смешных фрагментов, от самого смешного к менее смешному; для каждого одной фразой — в чём шутка
(если шутку не объяснить, фрагмент не бери). Для каждого — короткая подпись для экрана (до 28 символов, по-русски, без кавычек и хэштегов, с иронией; только про то, что
реально сказано во фрагменте — не выдумывай предметов и людей, которых в нём нет),
и общее название сборки (до 50 символов).

{items}""",
    'en': """Below are stream fragments (automatic transcript with errors — infer the meaning). Marks: [laugh] — laughter heard,
[scream] — loud reaction. Not funny: songs and music, in-game callouts, word salad, greetings and goodbyes.
Funny example: "I'm gonna destroy them all... oh. I'm dead" — a promise and an instant fail. Not funny: "two on B, he has AWP".
Pick the {k} funniest fragments, funniest first; for each say in one phrase what the joke is (if you can't, skip it). For each, a short on-screen caption (up to 32 characters,
no quotes or hashtags, ironic, only about what is actually said — invent nothing), and one compilation title (up to 50 characters).

{items}""",
}
EDITOR = {
    'ru': """Ты — главный редактор сборки «смешные моменты со стрима» в духе вирусных TikTok-нарезок: ирония, подколы, провалы,
реакции. Судьи уже отобрали финалистов. Выбери из них {k} лучших, от самого смешного (песни, игровые переговоры и бессвязное не бери),
для каждого одной фразой — в чём шутка, и подпись
для экрана (до 28 символов, ирония, без кавычек и хэштегов; только по словам фрагмента, ничего не выдумывай) и название всей сборки (до 50 символов).""",
    'en': """You are the chief editor of a viral "funny stream moments" compilation: irony, roasts, fails, reactions.
Judges picked the finalists. Choose the best {k}, funniest first (skip songs, callouts, word salad); for each say what the joke is and write an on-screen caption
(up to 32 characters, ironic, no quotes or hashtags) and a compilation title (up to 50 characters).""",
}

# Для обычных шортов (не сборки) — другая панель: цепляет ли начало, законченная ли мысль, досмотрит ли зритель.
VIRAL = {
    'judges': {
        'ru': {
            'крючок': """Ты — судья по КРЮЧКУ. Решает первая фраза: вопрос, спор, сильное утверждение, начало истории,
неожиданная эмоция — то, из-за чего не пролистывают ленту в первые 3 секунды.""",
            'история': """Ты — судья по ЗАВЕРШЁННОСТИ. Фрагмент — законченная мысль или история с развязкой: есть начало,
поворот и концовка (панчлайн, вывод, реакция). Обрывки и «середина разговора» — плохо.""",
            'зритель': """Ты — обычный зритель TikTok. Выбирай то, что досмотришь до конца и перешлёшь другу без контекста стрима:
смешно, иронично, неожиданно, эмоционально или полезно.""",
        },
        'en': {
            'hook': """You judge the HOOK: the first line — a question, a bold claim, a story opener, a sudden emotion — that stops the scroll in 3 seconds.""",
            'story': """You judge COMPLETENESS: a finished thought or story with a payoff (punchline, conclusion, reaction). Fragments are bad.""",
            'viewer': """You are a regular TikTok viewer. Pick what you'd watch to the end and send to a friend without stream context.""",
        },
    },
    'common': {
        'ru': """Ниже фрагменты стрима или видео (расшифровка автоматическая, с ошибками — додумывай смысл). Пометки: [смех] — в звуке
слышен смех, [крик] — громкая реакция. Плохо: песни и музыка, тактические переговоры в игре, бессвязный набор слов, приветствия донатам.
Выбери {k} лучших фрагментов для отдельных вертикальных роликов, от лучшего; для каждого одной фразой — чем цепляет
(если объяснить нечем, не бери), и заголовок-крючок для экрана (до 28 символов, по-русски, без кавычек и хэштегов,
только по тому, что реально сказано — ничего не выдумывай), и общую тему видео (до 50 символов).

{items}""",
        'en': """Below are fragments of a stream or video (automatic transcript with errors — infer meaning). Marks: [laugh], [scream].
Bad: songs, in-game callouts, word salad, donation greetings. Pick the {k} best fragments for standalone vertical clips, best first;
for each say in one phrase why it hooks (if you can't, skip it) and an on-screen hook title (up to 32 characters, no quotes or hashtags,
only about what is actually said — invent nothing), and one overall topic (up to 50 characters).

{items}""",
    },
    'editor': {
        'ru': """Ты — главный редактор вертикальных нарезок для TikTok, Reels и Shorts: ирония, истории, эмоции, сильные фразы.
Судьи отобрали финалистов. Выбери {k} лучших, от лучшего (песни, игровые переговоры и бессвязное не бери); для каждого одной фразой —
чем цепляет, и заголовок-крючок (до 28 символов, без кавычек и хэштегов, только по словам фрагмента), и общую тему (до 50 символов).""",
        'en': """You are the chief editor of vertical clips for TikTok, Reels and Shorts. Judges picked finalists. Choose the best {k}, best first
(skip songs, callouts, word salad); for each say why it hooks and write a hook title (up to 32 characters, only about what is said) and one topic.""",
    },
}
FUNNY = {'judges': JUDGES, 'common': COMMON, 'editor': EDITOR}
CTX_CHARS = 13000          # символов расшифровки на запрос: n_ctx 8192 − ответ 2000 − инструкции; по-русски ≈2.5 символа на токен


MAT = re.compile(r'(?<!\w)\w*(?:бля|блят|хуй|хуе|хуё|хуя|хуи|пизд|еба|ёба|ебу|ебл|ебн|сука|суки|гандон|гондон|залуп|пидор|пидар|мудак|мудил|дроч|шлюх|падла|паскуд)\w*', re.I)


def mat_share(text):
    """Доля нецензурных слов в тексте: сплошной мат — это оскорбления/ругань, а не шутка и не история."""
    w = re.findall(r'\w+', text)
    return sum(1 for x in w if MAT.fullmatch(x)) / max(1, len(w))


def clean(text, fallback=''):
    """Убирает нецензурные слова из подписи/заголовка (для публикации); пусто — fallback."""
    t = MAT.sub('', text)
    t = re.sub(r'\s+([,.;:!?])', r'\1', t); t = re.sub(r'([,.;:!?])(\s*[,.;:!?])+', r'\1', t)      # висящая пунктуация после вырезания
    t = ' '.join(t.split()).strip(' ,.;:—-–!?')
    if len(re.findall(r'\w+', t)) < 0.8 * len(re.findall(r'\w+', text)): return fallback      # мата было много — обрубок не годится
    return t if len(t) >= 3 else fallback


def _sung(text):
    t = text.lower()
    return len(re.findall(r'([аеиоуыэюяaeiouy])\1{4,}', t)) >= 2


def _verbatim(q, text, need=0.8):
    """Цитата действительно из текста: ≥80 % её слов (по первым 5 буквам) встречаются в расшифровке."""
    norm = lambda x: [w[:5] for w in re.findall(r'\w+', x.lower().replace('ё', 'е'))]
    qw, tw = norm(q), set(norm(text))
    return len(qw) >= 2 and sum(w in tw for w in qw) >= need * len(qw)


def _ctx(context, lg):
    """Строка-подсказка о видео (название стрима: кто в нём, о чём) — имена и тема берутся отсюда, а не из шумной расшифровки."""
    if not context: return ''
    return (f'О видео: «{context}». Имена людей и названия бери отсюда — в расшифровке они часто искажены.\n\n' if lg == 'ru'
            else f'About the video: "{context}". Take names from here — the transcript often garbles them.\n\n')


def _schema(k):
    return {'type': 'object', 'required': ['best', 'compilation_title'], 'properties': {
        'best': {'type': 'array', 'minItems': 1, 'maxItems': k, 'items': {
            'type': 'object', 'required': ['id', 'joke', 'title'],
            'properties': {'id': {'type': 'integer'}, 'joke': {'type': 'string', 'maxLength': 70},
                           'title': {'type': 'string', 'maxLength': 40}}}},
        'compilation_title': {'type': 'string', 'maxLength': 60}}}


def _ask(llm, system, user, k):
    try:
        r = llm.create_chat_completion(messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                                       temperature=0.2, max_tokens=2000, response_format={'type': 'json_object', 'schema': _schema(k)})
        return json.loads(r['choices'][0]['message']['content'])
    except Exception as e:
        _log(f'  (языковая модель ответила с ошибкой: {type(e).__name__}; голос пропущен)'); return None


def _clip(t, n=32):
    """Подпись не длиннее n символов, обрезка по слову."""
    t = ' '.join(t.strip().strip('"«»').split())
    if len(t) <= n: return t
    ws = t[:n + 1].rsplit(' ', 1)[0].split()
    while len(ws) > 1 and len(ws[-1].strip(',.:;—-–')) <= 2: ws.pop()       # не оставляем висящий предлог «на», «в»
    cut = ' '.join(ws).rstrip(',.:;—-– ')
    return cut if n // 2 <= len(cut) <= n else t[:n]               # без пробела в первых n + 1 символах cut длиннее n


def _grounded(title, text):
    """Подпись опирается на текст момента: хотя бы одно слово (по первым 4 буквам) есть в расшифровке."""
    stem = lambda w: w[:4]
    tw = {stem(w) for w in re.findall(r'\w{3,}', text.lower().replace('ё', 'е'))}
    return any(stem(w) in tw for w in re.findall(r'\w{3,}', title.lower().replace('ё', 'е')))


def _votes(data, ids, k):
    """[(номер кандидата, место, подпись)] из ответа; номера в ответе — позиции в показанном списке."""
    out, seen = [], set()
    for place, b in enumerate((data or {}).get('best', [])[:k]):
        j = b.get('id', 0) - 1
        if not 0 <= j < len(ids) or ids[j] in seen: continue
        seen.add(ids[j]); out.append((ids[j], place, _clip(b.get('title') or ''), (b.get('joke') or '').strip()))
    return out


def llm_rank(cands, k, lang, seed=0, repeats=1, mode='funny', context=None):
    """Панель судей + редактор (локальная Qwen3-4B); mode: 'funny' — смешное, 'viral' — обычные шорты. c['llm'] 0..1, c['votes'] — кто из судей выбрал. False — модели нет."""
    path = llm_path()
    if not path: return False
    try:
        from llama_cpp import Llama
    except ImportError:
        _log('  (для оценки смысла нужен llama-cpp-python: pip install -r requirements-funny.txt; оцениваю без неё)'); return False
    try:
        llm = Llama(model_path=path, n_ctx=8192, n_threads=max(1, (os.cpu_count() or 2)), verbose=False)
    except Exception as e:
        _log(f'  (языковая модель не загрузилась: {type(e).__name__}; оцениваю без неё)'); return False
    P = VIRAL if mode == 'viral' else FUNNY
    lg = lang if lang in P['judges'] else 'en'; width = 700 if mode == 'viral' else 400
    marks = ('[смех]', '[крик]') if lg == 'ru' else ('[laugh]', '[scream]')
    rnd = random.Random(seed); k = min(k, len(cands)); titles = {}; comp = []

    def listing(ids):                                  # кандидаты нумеруются заново для каждого судьи — в своём порядке
        lines = []; lim = min(width, max(80, CTX_CHARS // max(1, len(ids))))   # все кандидаты должны влезть в контекст
        for j, i in enumerate(ids):
            c = cands[i]; m = (f' {marks[0]}' if c.get('laugh_p', 0) > 0.06 else '') + (f' {marks[1]}' if c.get('scream_p', 0) > 0.1 else '')
            lines.append(f'{j + 1}.{m} {c["text"][:lim]}')
        return '\n'.join(lines)

    for c in cands: c['llm'] = 0.0; c['votes'] = []; c.pop('joke', None)
    judges = P['judges'][lg]; votes = len(judges) * repeats + repeats      # голос редактора весит как один судья
    for name, persona in judges.items():
        for r in range(repeats):                       # repeats > 1 — тот же судья ещё раз в другом порядке (стабильнее, дольше)
            ids = list(range(len(cands))); rnd.shuffle(ids)
            _log(f'    судья «{name}»…' if repeats == 1 else f'    судья «{name}» ({r + 1}/{repeats})…')
            for i, place, t, jk in _votes(_ask(llm, persona, _ctx(context, lg) + P['common'][lg].format(k=k, items=listing(ids)), k), ids, k):
                cands[i]['llm'] += (k - place) / k / votes
                if name not in cands[i]['votes']: cands[i]['votes'].append(name)
                if t: titles.setdefault(i, t)
                if jk: cands[i].setdefault('joke', jk)
    fin = sorted(range(len(cands)), key=lambda i: -cands[i]['llm'])[:min(len(cands), k + 4)]
    fin = [i for i in fin if cands[i]['llm'] > 0] or fin
    _log('    редактор выбирает итог…')
    data = _ask(llm, P['editor'][lg].format(k=k), _ctx(context, lg) + listing(fin), min(k, len(fin)))
    for i, place, t, jk in _votes(data, fin, k):
        cands[i]['llm'] += (k - place) / k * repeats / votes; cands[i]['votes'].append('редактор' if lg == 'ru' else 'editor')
        if t: titles[i] = t                            # подпись и объяснение редактора важнее
        if jk: cands[i]['joke'] = jk
    if data and data.get('compilation_title'): comp.append(_clip(data['compilation_title'], 50))
    for i, t in titles.items():
        if _grounded(t, cands[i]['text'] + ' ' + (context or '')): cands[i]['title'] = t; cands[i]['title_src'] = 'llm'   # выдуманную не берём
    m = max(c['llm'] for c in cands) or 1.0
    for c in cands: c['llm'] = round(c['llm'] / m, 3)          # 0..1: лучший по мнению судей = 1
    del llm
    return comp[0] if comp else True


def rank(cands, n, lang, load, retranscribe=None, cache=None, mode='funny', reserve=3, context=None):
    """cands — кандидаты эвристики (start, end, score). Возвращает n лучших по сумме оценок, по времени.
    mode='viral' — обычные шорты: другая панель судей, эвристика весит больше.
    cache — json с текстом и оценками звука по кандидатам: повторный запуск не распознаёт и не слушает заново."""
    if not cands: return cands
    label = 'смешные моменты' if mode == 'funny' else 'моменты'; viral = mode == 'viral'
    store = {}
    if cache and os.path.exists(cache):
        try: store = json.load(open(cache, encoding='utf-8'))
        except Exception: store = {}
    key = lambda c: f"{c['start']:.2f}-{c['end']:.2f}"
    todo = [c for c in cands if key(c) not in store]
    for c in cands:
        if key(c) in store: c.update(store[key(c)])
    if retranscribe and todo:
        _log(f'  {label}: точное распознавание {len(todo)} кандидатов…')
        for c in todo: c['text'] = retranscribe(c) or c.get('text', '')
    _log(f'  {label}: слушаю смех и крики (AST)…')
    have_ast = audio_scores(todo, load) if todo else all('laugh_p' in c for c in cands)
    if cache and have_ast:
        for c in cands: store[key(c)] = {k: c[k] for k in ('text', 'laugh_p', 'scream_p', 'sing_p') if k in c}
        tmp = cache + '.part'
        with open(tmp, 'w', encoding='utf-8') as f: json.dump(store, f, ensure_ascii=False)
        os.replace(tmp, cache)
    _log(f'  {label}: судьи выбирают лучшее…')
    comp = llm_rank(cands, min(n, len(cands)), lang, mode=mode, context=context)
    h = np.array([c['score'] for c in cands], float); h = (h - h.min()) / (np.ptp(h) + 1e-9)
    for c, hs in zip(cands, h):
        total = (0.6 if viral else 0.4) * hs                   # громкость сама по себе — не повод: решают судьи
        if comp and c['llm'] < 0.35: total -= 0.8              # за момент почти никто из судей не голосовал
        if _sung(c.get('text', '')): total -= 1.0              # тянущиеся «аааа/иииии» — пение под трек
        ms = mat_share(c.get('text', ''))
        if ms > 0.25: total -= 2.0                             # сплошная ругань / зачитывание оскорблений — не контент
        elif ms > 0.12: total -= 0.5 * (ms - 0.12) / 0.13 * 2
        if have_ast and viral: total += 0.4 * min(1.0, c['laugh_p'] / 0.15) - 1.5 * max(0.0, c['sing_p'] - 0.2)
        elif have_ast:                                         # вероятности AST на стримах малы (смех под музыкой) — шкала до 0.15
            total += 1.2 * min(1.0, c['laugh_p'] / 0.15) + 0.3 * min(1.0, c['scream_p'] / 0.15) - 1.5 * max(0.0, c['sing_p'] - 0.2)
        if comp: total += 2.0 * c['llm']
        c['funny'] = round(float(total), 3)
        c['why'] = ((f'{c["joke"]}; ' if c.get('joke') else f'{c["read_why"]}; ' if c.get('read_why') else '')
                    + ('нашёл читатель; ' if c.get('read_why') else '') + f'эвристика {hs:.2f}' + (f', смех {c["laugh_p"]:.2f}, крик {c["scream_p"]:.2f}' if have_ast else '')
                    + (f', судьи {c["llm"]:.2f} ({"/".join(c["votes"]) or "—"})' if comp else ''))
    for c in cands:                                      # без подписи судей — начало фразы из точной расшифровки
        if c.get('title_src') != 'llm' and c.get('text'):
            c['title'] = _clip(re.split(r'(?<=[.!?…])\s', c['text'].strip())[0].rstrip('.!?…'))
        c['title'] = _clip(clean(c.get('title', ''), 'Момент со стрима'))   # заголовок для экрана — без мата
    order = sorted(cands, key=lambda c: -c['funny']); best = order[:n]
    short = 30.0                                          # в пятёрке нужны и короткие ролики, не только минутные
    if mode == 'viral' and n >= 3 and all(c['end'] - c['start'] > short for c in best):
        alt = [c for c in order[n:n * 3] if c['end'] - c['start'] <= short and c['funny'] >= 0.6 * best[-1]['funny']]
        if alt:
            alt[0]['why'] += '; контролёр: короткий ролик для разнообразия'; order.remove(alt[0]); best = best[:-1] + [alt[0]]
            order = best + [c for c in order if c not in best]
    spare = [dict(c, reserve=True) for c in order[n:n + reserve]]          # запас для контролёра: заменит брак
    if isinstance(comp, str):
        for c in best + spare: c['compilation_title'] = comp
    return sorted(best, key=lambda c: c['start']) + spare


POST = {
    'ru': """Ты — SMM-редактор TikTok, Reels и Shorts. Для каждого фрагмента стрима ниже напиши пост:
title — заголовок-крючок для экрана (до 28 символов, без кавычек и хэштегов): интрига или суть момента («Позвонил врагу в прямом эфире»,
«Последние 100 рублей на лайки»), а не пересказ случайной фразы; непонятные слова из расшифровки (ошибки распознавания) не используй;
quote — самая сильная фраза момента, ДОСЛОВНО скопированная из фрагмента (до 90 символов) — она станет подписью к посту;
hashtags — 5–7 хэштегов без символа #: тема, имена гостей и стримера (как в названии видео), общие (стрим, юмор, fyp);
не бери обрывки и непонятные слова из расшифровки;
Только по тому, что реально сказано во фрагменте — ничего не выдумывай.

{items}""",
    'en': """You are a TikTok/Reels/Shorts SMM editor. For each stream fragment below write a post:
title — on-screen hook title (up to 32 characters, no quotes or hashtags): the intrigue or gist, not a random quote; skip garbled words;
quote — the strongest line of the moment copied VERBATIM from the fragment (up to 90 characters) — it becomes the post caption;
hashtags — 5–7 hashtags without #. Only about what is actually said — invent nothing.

{items}""",
}


def describe(clips, lang, context=None):
    """Описание и хэштеги к каждому готовому моменту (локальная модель). False — модели нет."""
    clips = [c for c in clips if c.get('text')]
    path = llm_path(download=False)
    if not clips or not path: return False
    try:
        from llama_cpp import Llama
    except ImportError:
        return False
    lg = lang if lang in POST else 'en'
    schema = {'type': 'object', 'required': ['posts'], 'properties': {'posts': {'type': 'array', 'items': {
        'type': 'object', 'required': ['id', 'title', 'quote', 'hashtags'],
        'properties': {'id': {'type': 'integer'}, 'title': {'type': 'string', 'maxLength': 40}, 'quote': {'type': 'string', 'maxLength': 110},
                       'hashtags': {'type': 'array', 'minItems': 3, 'maxItems': 8, 'items': {'type': 'string', 'maxLength': 30}}}}}}}
    _log('  пишу описания и хэштеги к постам…')
    try:
        llm = Llama(model_path=path, n_ctx=8192, n_threads=max(1, (os.cpu_count() or 2)), verbose=False)
    except Exception as e:
        _log(f'  (описания не получились: {type(e).__name__})'); return False
    done = False
    for b in range(0, len(clips), 8):                  # по 8 постов за запрос: и вопрос, и ответ влезают в контекст
        part = clips[b:b + 8]
        items = '\n'.join(f'{i + 1}. [{c.get("title", "")}] {c["text"][:600]}' for i, c in enumerate(part))
        try:
            r = llm.create_chat_completion(messages=[{'role': 'user', 'content': _ctx(context, lg) + POST[lg].format(items=items)}], temperature=0.4,
                                           max_tokens=2500, response_format={'type': 'json_object', 'schema': schema})
            data = json.loads(r['choices'][0]['message']['content'])
        except Exception as e:
            _log(f'  (описания не получились: {type(e).__name__})'); continue
        for p in data.get('posts', []):
            i = p.get('id', 0) - 1
            if 0 <= i < len(part):
                q = ' '.join(p.get('quote', '').split()).strip('"«»„“ ')
                if q and _verbatim(q, part[i]['text']) and not MAT.search(q):   # описание — настоящая цитата из ролика, без мата
                    part[i]['description'] = f'«{q}»'
                else: part[i]['description'] = clean(part[i].get('title', ''))
                t = _clip(p.get('title') or '')
                if t and part[i].get('title_src') != 'llm' and _grounded(t, part[i]['text'] + ' ' + (context or '')):   # судьи подпись не дали — берём отсюда
                    part[i]['title'] = clean(t, part[i].get('title', '')); part[i]['title_src'] = 'llm'
                tags = []
                for t in p.get('hashtags', []):
                    t = re.sub(r'[^\w]', '', t.lstrip('#'))
                    if t and t.lower() not in {x.lower() for x in tags}: tags.append(t)
                part[i]['hashtags'] = tags[:8]
                done = True
    del llm
    return done


READER = {
    'ru': """Ты — монтажёр вертикальных нарезок (TikTok, Reels, Shorts). Ниже кусок расшифровки стрима, по строке на фразу,
с номером строки (расшифровка автоматическая, с ошибками — додумывай смысл). Найди до {m} лучших моментов для отдельных роликов:
{what}. Момент — подряд идущие строки длиной от {dmin:.0f} до {dmax:.0f} секунд (время строки в скобках); длина — по содержанию:
короткая шутка, реакция или сильная фраза — 15–25 с, история или спор — 30–60 с; лишнего не прихватывай,
начинается с подводки (чтобы было понятно без контекста), заканчивается панчлайном или реакцией.
Не бери: песни и музыку, приветствия донатам, технические разговоры, бессвязный текст. Если хороших моментов нет — верни пустой список.
Для каждого: first и last — номера первой и последней строки, why — одной фразой, чем цепляет.

{lines}""",
    'en': """You edit vertical clips (TikTok, Reels, Shorts). Below is a chunk of a stream transcript, one sentence per numbered line
(automatic, with errors). Find up to {m} best moments for standalone clips: {what}. A moment is consecutive lines of about
{dmin:.0f}–{dmax:.0f} seconds (line time in brackets) — length by content: a quick joke or reaction 15–25 s, a story 30–60 s;
starts with the setup, ends with the punchline or reaction.
Skip songs, donation greetings, technical talk, word salad. If nothing is good, return an empty list.
For each: first and last line numbers, why — one phrase on what hooks.

{lines}""",
}
WHAT = {'funny': {'ru': 'смешное и ироничное — шутки, подколы, провалы, абсурд, бурные реакции',
                  'en': 'funny and ironic — jokes, roasts, fails, absurd moments, wild reactions'},
        'viral': {'ru': 'интересное и цепляющее — истории с развязкой, споры, откровения, сильные фразы, а также смешное и ироничное',
                  'en': 'interesting and hooking — stories with a payoff, arguments, confessions, strong lines, and funny or ironic bits'}}


def read_transcript(sents, lang, dmin, dmax, mode='viral', chunk=600.0, per_chunk=3, context=None):
    """Агент-«читатель»: читает всю расшифровку кусками по chunk секунд, как монтажёр, и предлагает моменты по смыслу —
    то, что эвристика по словам и громкости пропускает (истории, откровения). sents — фразы [[{'w','s','e'}…]…]."""
    path = llm_path()
    if not path or not sents: return []
    try:
        from llama_cpp import Llama
    except ImportError:
        return []
    lg = lang if lang in READER else 'en'
    schema = {'type': 'object', 'required': ['moments'], 'properties': {'moments': {'type': 'array', 'maxItems': per_chunk, 'items': {
        'type': 'object', 'required': ['first', 'last', 'why'],
        'properties': {'first': {'type': 'integer'}, 'last': {'type': 'integer'}, 'why': {'type': 'string', 'maxLength': 90}}}}}}
    try:
        llm = Llama(model_path=path, n_ctx=8192, n_threads=max(1, (os.cpu_count() or 2)), verbose=False)
    except Exception as e:
        _log(f'  (языковая модель не загрузилась: {type(e).__name__}; читатель пропущен)'); return []
    out, i, total = [], 0, sents[-1][-1]['e']
    while i < len(sents):
        t_end = sents[i][0]['s'] + chunk; j = i
        while j < len(sents) and sents[j][0]['s'] < t_end: j += 1
        part = sents[i:j]
        _log(f'    читатель: {sents[i][0]["s"] / 60:.0f}–{min(t_end, total) / 60:.0f} мин…')
        lines = '\n'.join(f'{k + 1}. [{s[0]["s"]:.0f} с] ' + ' '.join(w['w'] for w in s)[:300] for k, s in enumerate(part))
        msg = _ctx(context, lg) + READER[lg].format(m=per_chunk, what=WHAT[mode][lg], dmin=dmin, dmax=dmax, lines=lines)
        try:
            r = llm.create_chat_completion(messages=[{'role': 'user', 'content': msg}], temperature=0.2, max_tokens=600,
                                           response_format={'type': 'json_object', 'schema': schema})
            data = json.loads(r['choices'][0]['message']['content'])
        except Exception as e:
            _log(f'    (читатель ошибся: {type(e).__name__}; кусок пропущен)'); data = {}
        for m in data.get('moments', []):
            a, b = m.get('first', 0) - 1, m.get('last', 0) - 1
            if not (0 <= a <= b < len(part)): continue
            while b > a and part[b][-1]['e'] - part[a][0]['s'] > dmax: b -= 1          # длиннее нужного — режем концовку,
            while b + 1 < len(part) and part[b][-1]['e'] - part[a][0]['s'] < dmin: b += 1   # короче — дотягиваем фразами
            ws = [w for s in part[a:b + 1] for w in s]
            if ws[-1]['e'] - ws[0]['s'] < dmin * 0.6: continue
            out.append({'start': ws[0]['s'], 'end': ws[-1]['e'] + 0.6, 'score': 0.0, 'text': ' '.join(w['w'] for w in ws),
                        'title': '', 'hook': '', 'description': '', 'hashtags': [], 'emphasis': [],
                        'read_why': (m.get('why') or '').strip()})
        i = max(j, i + 1)
    del llm
    return out
