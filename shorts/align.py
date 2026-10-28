"""Точная привязка слов к звуку (forced alignment, как в WhisperX).

Whisper хорошо понимает, ЧТО сказано, но время слов у него «плавает» на 0.3–1 с (особенно у turbo).
Здесь текст от whisper выравнивается по звуку акустической моделью wav2vec2 (CTC): для каждой буквы
находится кадр 20 мс, где она звучит, и слово получает точные начало и конец.
"""
import re
import numpy as np

MODELS = {'ru': 'jonatasgrosman/wav2vec2-large-xlsr-53-russian', 'en': 'jonatasgrosman/wav2vec2-large-xlsr-53-english'}
_cache = {}
LAT = dict(zip('abcdefghijklmnopqrstuvwxyz', ['а', 'б', 'к', 'д', 'е', 'ф', 'г', 'х', 'и', 'дж', 'к', 'л', 'м', 'н', 'о', 'п', 'к', 'р', 'с', 'т', 'у', 'в', 'в', 'кс', 'й', 'з']))
LEAD = 0.06          # замерено: wav2vec2 даёт начало на ~60 мс позже — субтитр лучше чуть раньше звука


def _load(lang):
    if lang not in _cache:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, logging as tlog
        tlog.set_verbosity_error(); tlog.disable_progress_bar()
        name = MODELS[lang]
        proc = Wav2Vec2Processor.from_pretrained(name); model = Wav2Vec2ForCTC.from_pretrained(name).eval()
        torch.set_num_threads(max(1, torch.get_num_threads()))
        _cache[lang] = (proc, model)
    return _cache[lang]


def _emissions(audio, lang, sr=16000, chunk=30):
    import torch
    proc, model = _load(lang); out = []
    for i in range(0, len(audio), chunk * sr):                 # кусками по 30 с, чтобы не съесть память
        piece = audio[i:i + chunk * sr]; real = len(piece)
        if real < 8000: piece = np.pad(piece, (0, 8000 - real))  # короткий хвост: модель не принимает < 25 мс
        x = proc(piece, sampling_rate=sr, return_tensors='pt').input_values
        with torch.inference_mode():
            em = torch.log_softmax(model(x).logits[0], dim=-1).numpy()
        out.append(em[:max(1, int(round(em.shape[0] * real / len(piece))))])
    return np.concatenate(out), proc.tokenizer.get_vocab()


def _viterbi(em, tokens, blank):
    """CTC-выравнивание: путь через буквы с максимальной вероятностью → кадр каждой буквы."""
    T, N = em.shape[0], len(tokens)
    tr = np.full((T + 1, N + 1), -np.inf, np.float32); tr[0, 0] = 0
    tr[1:, 0] = np.cumsum(em[:, blank])
    for t in range(T):
        stay = tr[t, 1:] + em[t, blank]; move = tr[t, :-1] + em[t, tokens]
        tr[t + 1, 1:] = np.maximum(stay, move)
    # обратный проход
    j = N; t = T; path = []
    while j > 0 and t > 0:
        stay = tr[t - 1, j] + em[t - 1, blank]; move = tr[t - 1, j - 1] + em[t - 1, tokens[j - 1]]
        if move >= stay: path.append((j - 1, t - 1)); j -= 1
        t -= 1
    return path[::-1]


def align(words, audio, offset, lang='ru', sr=16000):
    """words: [{'w','s','e'}] (время от начала видео), audio — кусок, начинающийся в offset.
    Возвращает words с точным временем или None, если выровнять нельзя (язык не поддержан, нет модели)."""
    if lang not in MODELS or not words: return None
    try:
        em, vocab = _emissions(audio, lang, sr)
    except Exception as e:                                       # нет torch/transformers/сети — вызывающий возьмёт запасной вариант
        print(f'  (точная привязка слов недоступна: {type(e).__name__}: {str(e)[:120]}; беру запасной способ)'); return None
    blank = vocab.get('<pad>', 0); sep = vocab.get('|')
    stride = len(audio) / sr / em.shape[0]
    tokens, owner = [], []
    for wi, w in enumerate(words):
        t = w['w'].lower().replace('ё', 'е') if 'ё' not in vocab else w['w'].lower()
        if lang == 'ru': t = ''.join(LAT.get(ch, ch) for ch in t)       # AWP → авп: латиница тоже выравнивается
        for ch in t:
            if ch in vocab: tokens.append(vocab[ch]); owner.append(wi)
        if sep is not None: tokens.append(sep); owner.append(-1)
    if not tokens or len(tokens) > em.shape[0]: return None
    path = _viterbi(em, np.array(tokens), blank)
    first, last = {}, {}
    for j, t in path:
        wi = owner[j]
        if wi < 0: continue
        first.setdefault(wi, t); last[wi] = t
    out = []
    for wi, w in enumerate(words):
        w = dict(w)
        if wi in first:
            w['s'] = round(offset + first[wi] * stride - LEAD, 3); w['e'] = round(offset + (last[wi] + 1) * stride, 3)
            w['e'] = min(w['e'], w['s'] + 1.5)                    # растянутое через паузу слово не держим долго
        out.append(w)
    done = [wi in first for wi in range(len(out))]               # слова без букв модели (цифры, латиница) — между соседями
    for i, w in enumerate(out):
        if done[i]: continue
        lo = next((out[j]['e'] for j in range(i - 1, -1, -1) if done[j]), None)
        hi = next((out[j]['s'] for j in range(i + 1, len(out)) if done[j]), None)
        if lo is not None and hi is not None and hi > lo: w['s'], w['e'] = round(lo, 3), round(hi, 3)
        elif lo is not None: w['s'] = max(w['s'], lo)
        elif hi is not None: w['s'] = min(w['s'], hi - 0.1); w['e'] = min(w['e'], hi)   # первое слово клипа без букв модели
    for i in range(1, len(out)):                                 # монотонность и s < e
        if out[i]['s'] < out[i - 1]['s']: out[i]['s'] = out[i - 1]['s']
    for w in out:
        if w['e'] <= w['s']: w['e'] = round(w['s'] + 0.1, 3)
    return out
