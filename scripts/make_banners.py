"""Draw the notebook header banners and the "Supported by Reflective" badge (PNG, 2x resolution).

    uv run --no-project --with pillow python scripts/make_banners.py --brand <Reflective brand assets>

The brand folder is the Reflective asset pack (01_Logo, 02_Fonts with Gambarino and Open Sans);
its fonts are not in this repository, the PNGs it makes are (assets/). Images look the same in
Colab, JupyterLab, VS Code, GitHub and in dark themes, unlike HTML styles in markdown cells,
which Colab strips. The notebooks link the PNGs at a fixed commit (config.ASSET_COMMIT), so
after regenerating them, commit them first and then update ASSET_COMMIT.
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SCALE = 2                                   # pixels per CSS pixel (sharp on high-DPI screens)

# Reflective palette (brand assets, 03_Colors). Yellow is never used for text.
BLUE, GREY_MID, GOLD = (70, 111, 141), (147, 146, 128), (164, 127, 82)     # brand gradient stops
BEIGE_50, BEIGE_10, BEIGE = (245, 236, 228), (254, 253, 252), (225, 203, 186)
CHARCOAL, SLATE = (40, 44, 46), (139, 141, 142)
WHITE = (255, 255, 255)

BANNER_W, BANNER_H, RADIUS, PAD = 1200, 300, 18, 56                         # CSS pixels
KICKER = "AAAR 2026  ·  Photochemical box modeling tutorial"

# One banner per notebook; add the others here when they get the design.
BANNERS = {
    "banner-level1.png": {"title": "Level 1 · Volatility",
                          "subtitle": "When does a vapor make particles grow?", "level": 1},
}
N_LEVELS = 3


def px(v: float) -> int:
    return round(v * SCALE)


def font(brand: Path, name: str, size_css: float) -> ImageFont.FreeTypeFont:
    paths = {"title": "02_Fonts/Gambarino_Complete/Fonts/OTF/Gambarino-Regular.otf",
             "regular": "02_Fonts/OpenSans/OpenSans-Regular.ttf",
             "medium": "02_Fonts/OpenSans/OpenSans-Medium.ttf",
             "semibold": "02_Fonts/OpenSans/OpenSans-SemiBold.ttf"}
    return ImageFont.truetype(str(brand / paths[name]), px(size_css))


def logo(brand: Path, colour: str, height_css: float, monogram: bool = False) -> Image.Image:
    """Official RGB logo (or monogram), cropped to its ink and scaled to a height."""
    folder, stem = ("Monogram-Icon", "Monogram") if monogram else ("Logo", "Logo")
    im = Image.open(brand / f"01_Logo/RBG (Screen)/{folder}/PNG/{stem}-{colour}.png").convert("RGBA")
    im = im.crop(im.getchannel("A").getbbox())
    h = px(height_css)
    return im.resize((round(im.width * h / im.height), h), Image.LANCZOS)


def gradient(w: int, h: int) -> Image.Image:
    """Horizontal brand gradient Blue -> #939280 -> Gold."""
    row = Image.new("RGB", (w, 1))
    for x in range(w):
        t = x / (w - 1)
        a, b, u = (BLUE, GREY_MID, t / 0.5) if t <= 0.5 else (GREY_MID, GOLD, (t - 0.5) / 0.5)
        row.putpixel((x, 0), tuple(round(a[i] + (b[i] - a[i]) * u) for i in range(3)))
    return row.resize((w, h))


def rounded(im: Image.Image, radius_css: float) -> Image.Image:
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, im.width - 1, im.height - 1), px(radius_css), fill=255)
    out = im.convert("RGBA")
    out.putalpha(mask)
    return out


def tracked(draw: ImageDraw.ImageDraw, xy, text: str, fnt, fill, tracking_css: float) -> None:
    """Text with letter spacing (for the small-caps kicker)."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += draw.textlength(ch, font=fnt) + px(tracking_css)


def banner(brand: Path, title: str, subtitle: str, level: int) -> Image.Image:
    w, h = px(BANNER_W), px(BANNER_H)
    im = gradient(w, h).convert("RGBA")

    mark = logo(brand, "White", 330, monogram=True)                  # faint watermark
    mark.putalpha(mark.getchannel("A").point(lambda a: a * 0.10))
    im.alpha_composite(mark, (w - mark.width + px(40), (h - mark.height) // 2))

    d = ImageDraw.Draw(im)
    tracked(d, (px(PAD), px(46)), KICKER.upper(), font(brand, "semibold", 13), BEIGE_50, 2.2)
    d.text((px(PAD - 3), px(72)), title, font=font(brand, "title", 64), fill=WHITE)
    d.text((px(PAD), px(162)), subtitle, font=font(brand, "regular", 24), fill=BEIGE_10)

    y_dots, r = px(BANNER_H - 50), px(5.5)                                   # progress: Level k of N
    for k in range(N_LEVELS):
        cx = px(PAD + 6) + k * px(22)
        box = (cx - r, y_dots - r, cx + r, y_dots + r)
        d.ellipse(box, fill=WHITE if k < level else None, outline=WHITE, width=px(1.5))
    d.text((px(PAD + 6) + N_LEVELS * px(22), y_dots), f"Level {level} of {N_LEVELS}",
           font=font(brand, "medium", 14), fill=BEIGE_50, anchor="lm")

    white = logo(brand, "White", 34)                                         # Supported by Reflective
    x_logo, y_logo = w - px(PAD) - white.width, h - px(40) - white.height
    im.alpha_composite(white, (x_logo, y_logo))
    d.text((x_logo + px(2), y_logo - px(10)), "Supported by", font=font(brand, "medium", 13),
           fill=BEIGE_50, anchor="ls")
    return rounded(im, RADIUS)


def badge(brand: Path) -> Image.Image:
    """White pill: "Supported by" + black logo; readable on light and dark backgrounds."""
    fnt, black = font(brand, "medium", 13), logo(brand, "Black", 26)
    label_w = round(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength("Supported by", font=fnt))
    pad, gap = px(16), px(12)
    w, h = pad + label_w + gap + black.width + pad, px(56)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), px(12), fill=WHITE, outline=BEIGE, width=px(1.5))
    d.text((pad, h // 2), "Supported by", font=fnt, fill=SLATE, anchor="lm")
    im.alpha_composite(black, (pad + label_w + gap, (h - black.height) // 2))
    return im


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brand", type=Path, required=True, help="Reflective brand assets folder")
    args = parser.parse_args()
    out = ROOT / "assets"
    out.mkdir(exist_ok=True)
    for name, spec in BANNERS.items():
        banner(args.brand, **spec).save(out / name, optimize=True)
        print(f"wrote assets/{name}")
    badge(args.brand).save(out / "supported-by-reflective.png", optimize=True)
    print("wrote assets/supported-by-reflective.png")


if __name__ == "__main__":
    main()
