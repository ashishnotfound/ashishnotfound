from pathlib import Path
from PIL import Image
import numpy as np
import cv2, shutil

ROOT = Path('reyo-wall-art/input')
OUT = Path('reyo-wall-art/images')
QUADS = {
    'wall': [(527.4,184.4),(856.9,184.0),(855.9,732.3),(527.0,729.9)],
    'bedroom': [(475.2,160.1),(758.9,161.1),(758.0,631.9),(474.6,631.0)],
    'desk': [(375.6,163.6),(767.3,162.8),(766.9,815.7),(374.1,813.8)],
    'closeup': [(426.3,-4.0),(924.7,-4.3),(924.0,826.1),(425.9,822.2)],
    'a4_size': [(184.0,420.3),(795.0,420.5),(795.4,1434.9),(184.0,1436.0)],
}

def contain(img, w, h):
    canvas = Image.new('RGB', (w, h), 'white')
    img = img.convert('RGB')
    scale = min(w / img.width, h / img.height)
    nw, nh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas.paste(resized, ((w-nw)//2, (h-nh)//2))
    return canvas

def overlay(template, art, quad):
    base = np.array(template.convert('RGB'))
    q = np.array(quad, dtype=np.float32)
    tw = int(round(max(np.linalg.norm(q[1]-q[0]), np.linalg.norm(q[2]-q[3]))))
    th = int(round(max(np.linalg.norm(q[3]-q[0]), np.linalg.norm(q[2]-q[1]))))
    patch = contain(art, max(1, tw), max(1, th))
    src = np.array([[0,0],[tw-1,0],[tw-1,th-1],[0,th-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, q)
    warped = cv2.warpPerspective(np.array(patch), H, (base.shape[1], base.shape[0]), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    mask = cv2.warpPerspective(np.full((th,tw),255,np.uint8), H, (base.shape[1],base.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    base[mask > 0] = warped[mask > 0]
    return Image.fromarray(base)

def main_image(art):
    canvas = Image.new('RGB', (1200,1600), 'white')
    page = contain(art, 920, 1300)
    canvas.paste(page, ((1200-920)//2, (1600-1300)//2))
    return canvas

def save(im, path, quality=86):
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, 'JPEG', quality=quality, optimize=True, progressive=True, subsampling=1)

if OUT.exists():
    shutil.rmtree(OUT)
for i in range(1, 51):
    art = Image.open(ROOT / 'artworks' / f'{i:03d}.webp')
    dest = OUT / f'REYO-A4-{i:03d}'
    save(main_image(art), dest / 'main.jpg', 90)
    for name in ['wall','bedroom','desk','closeup','a4_size']:
        template = Image.open(ROOT / 'templates' / f'{name}.webp')
        filename = 'a4-size.jpg' if name == 'a4_size' else f'{name}.jpg'
        save(overlay(template, art, QUADS[name]), dest / filename, 86)
print('Generated', sum(1 for _ in OUT.rglob('*.jpg')), 'images')
