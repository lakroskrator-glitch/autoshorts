"""Кино-оформление: цветокор, виньетка, зерно, «панчи» зума на акцентах, частицы поверх кадра."""
import math, random, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

GRADES = {
    'none': '',
    # тил-оранж, контраст, чуть сочнее, виньетка, резкость, мелкое зерно
    'cinema': ('eq=contrast=1.12:saturation=1.18:gamma=0.96,'
               'colorbalance=rs=-0.06:gs=0.0:bs=0.08:rm=0.02:bm=-0.02:rh=0.07:gh=0.02:bh=-0.06,'
               'vignette=PI/5,unsharp=5:5:0.6,noise=alls=6:allf=t'),
    'cold': 'eq=contrast=1.1:saturation=0.9,colorbalance=rs=-0.05:bs=0.1:rh=-0.03:bh=0.05,vignette=PI/5,noise=alls=5:allf=t',
    'warm': 'eq=contrast=1.08:saturation=1.2,colorbalance=rs=0.06:bs=-0.05:rh=0.08:bh=-0.08,vignette=PI/5,noise=alls=5:allf=t',
}


def punch_scale(times, W, H, amount=0.09, width=0.18):
    """Фильтр «удар зумом» в моменты times (с): кадр на долю секунды увеличивается и плавно возвращается."""
    if not times: return ''
    terms = '+'.join(f'{amount}*exp(-pow((t-{t:.2f})/{width},2))' for t in times[:40])
    z = f'(1+{terms})'
    return f"scale=w='trunc({W}*{z}/2)*2':h='trunc({H}*{z}/2)*2':eval=frame,crop={W}:{H}"


def particles(path, dur, W, H, fps=30, n=70, seed=1):
    """Видео частиц (светлые боке и пылинки на чёрном) — накладывается режимом screen."""
    rnd = random.Random(seed)
    ps = [{'x': rnd.uniform(0, W), 'y': rnd.uniform(0, H), 'r': rnd.choice([2, 3, 4, 6, 10, 16]), 'vx': rnd.uniform(-15, 15),
           'vy': rnd.uniform(15, 70), 'a': rnd.uniform(0.25, 0.9), 'ph': rnd.uniform(0, 6.28)} for _ in range(n)]
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
                           '-c:v', 'libx264', '-crf', '22', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', path], stdin=subprocess.PIPE)
    for f in range(int(dur * fps)):
        im = Image.new('RGB', (W // 2, H // 2)); dr = ImageDraw.Draw(im); t = f / fps
        for p in ps:
            x = (p['x'] + p['vx'] * t) % W / 2; y = (p['y'] + p['vy'] * t) % H / 2; r = p['r'] / 2
            v = int(255 * p['a'] * (0.6 + 0.4 * math.sin(t * 2 + p['ph'])))
            dr.ellipse([x - r, y - r, x + r, y + r], fill=(v, int(v * 0.95), int(v * 0.85)))
        im = im.filter(ImageFilter.GaussianBlur(1.6)).resize((W, H))
        ff.stdin.write(im.tobytes())
    ff.stdin.close(); ff.wait()
    return path
