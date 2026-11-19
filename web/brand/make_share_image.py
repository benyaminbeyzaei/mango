"""Make web/public/og-image.png, the 1200x630 picture shown when a link to the site is shared.

Run from the checkout root: python3 web/brand/make_share_image.py web/brand/mango.png web/public FONT.ttf
(needs Pillow; FONT is any bold sans font file, e.g. /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf)."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

src, out, font = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
CREAM, INK, MUTED = (255, 246, 214), (40, 32, 20), (110, 92, 60)

canvas = Image.new("RGB", (1200, 630), CREAM)
logo = Image.open(src).convert("RGBA")
logo = logo.crop(logo.getchannel("A").getbbox())
logo.thumbnail((440, 440), Image.LANCZOS)
canvas.paste(logo, (90, (630 - logo.height) // 2), logo)

draw = ImageDraw.Draw(canvas)
x = 600
draw.text((x, 170), "Mango", font=ImageFont.truetype(font, 104), fill=INK)
small = ImageFont.truetype(font, 34)
for i, line in enumerate(["Spoken question answering:", "retrieval methods compared", "on quality and cost"]):
    draw.text((x, 320 + i * 50), line, font=small, fill=MUTED)

canvas.save(out / "og-image.png", optimize=True)
print(f"og-image.png {(out / 'og-image.png').stat().st_size / 1024:.1f} KB")
