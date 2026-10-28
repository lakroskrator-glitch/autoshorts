"""Залипательный фон для нижней части (как в нарезках): шары прыгают внутри кольца и тянут линии к точкам отскока."""
import colorsys, math, random, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render(path, dur, w=1080, h=1230, fps=30, seed=None, font=None):
    rnd = random.Random(seed); cx, cy, R = w / 2, h / 2, min(w, h) * 0.44
    balls = []
    for k in range(3):
        hue = (k / 3 + rnd.uniform(-0.05, 0.05)) % 1
        balls.append({'p': np.array([cx + rnd.uniform(-80, 80), cy - R * 0.4]), 'v': np.array([rnd.uniform(-500, 500), 0.0]),
                      'r': 26.0, 'hue': hue, 'hits': []})
    g = np.array([0, 1500.0]); dt = 1 / fps; sub = 4; bounces = 0
    try:
        f = ImageFont.truetype(font, 64) if font else ImageFont.load_default()
    except OSError:
        f = ImageFont.load_default()
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
                           '-c:v', 'libx264', '-crf', '20', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', path], stdin=subprocess.PIPE)
    for _ in range(int(dur * fps)):
        for _ in range(sub):
            for b in balls:
                b['v'] = b['v'] + g * dt / sub; b['p'] = b['p'] + b['v'] * dt / sub
                d = b['p'] - [cx, cy]; dist = np.linalg.norm(d)
                if dist + b['r'] > R:                          # отскок от кольца с небольшим ускорением
                    n = d / dist; b['p'] = np.array([cx, cy]) + n * (R - b['r'])
                    b['v'] = (b['v'] - 2 * np.dot(b['v'], n) * n) * 1.012
                    sp = np.linalg.norm(b['v'])
                    if sp > 2600: b['v'] *= 2600 / sp
                    b['hits'].append(tuple(np.array([cx, cy]) + n * R)); b['hits'] = b['hits'][-60:]; bounces += 1
        im = Image.new('RGB', (w, h), (0, 0, 0)); dr = ImageDraw.Draw(im)
        dr.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(240, 240, 240), width=6)
        for b in balls:
            col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(b['hue'], 0.75, 1))
            for q in b['hits']:
                dr.line([tuple(b['p']), q], fill=col, width=4)
            r = b['r']; dr.ellipse([b['p'][0] - r, b['p'][1] - r, b['p'][0] + r, b['p'][1] + r], fill=col, outline=(255, 255, 255), width=3)
        dr.text((cx, cy), str(bounces), fill=(90, 90, 90), font=f, anchor='mm')
        for b in balls: b['hue'] = (b['hue'] + 0.002) % 1
        ff.stdin.write(im.tobytes())
    ff.stdin.close(); ff.wait()
    return path
