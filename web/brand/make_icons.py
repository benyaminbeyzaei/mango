"""Make every Mango icon in web/public from the source PNG (transparent background).

Run from the checkout root: python3 web/brand/make_icons.py web/brand/mango.png web/public (needs Pillow)."""
import sys
from pathlib import Path
from PIL import Image, ImageFilter

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
CREAM = (255, 246, 214, 255)  # opaque background where a platform needs one

im = Image.open(src).convert("RGBA")
im = im.crop(im.getchannel("A").getbbox())

def square(img, pad=0.0, bg=(0, 0, 0, 0)):
    """Centre on a square canvas; pad is the margin on each side as a share of the side."""
    side = round(max(img.size) / (1 - 2 * pad))
    canvas = Image.new("RGBA", (side, side), bg)
    canvas.alpha_composite(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas

def size(img, n, sharpen=False):
    small = img.resize((n, n), Image.LANCZOS)
    return small.filter(ImageFilter.UnsharpMask(radius=0.6, percent=60, threshold=1)) if sharpen else small

def save_png(img, name, colours=None):
    if colours:  # palette PNG: much smaller, fine for flat illustrations
        img = img.quantize(colours, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.FLOYDSTEINBERG)
    img.save(out / name, optimize=True)

tight = square(im, 0.02)
save_png(size(tight, 32, sharpen=True), "favicon-32.png")
size(tight, 48).save(out / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
save_png(size(square(im, 0.10, CREAM), 180), "apple-touch-icon.png", 256)
save_png(size(square(im, 0.04), 192), "icon-192.png", 256)
save_png(size(square(im, 0.04), 512), "icon-512.png", 256)
save_png(size(square(im, 0.20, CREAM), 512), "icon-maskable-512.png", 256)  # Android crops to a circle: keep 20 % safe zone
save_png(size(tight, 64), "logo-64.png", 256)     # header, shown at 32 px (2x for sharp screens)
save_png(size(tight, 160), "logo-160.png", 256)   # home page, shown at 80 px
for f in sorted(out.iterdir()):
    print(f"{f.name:24} {f.stat().st_size / 1024:6.1f} KB")
