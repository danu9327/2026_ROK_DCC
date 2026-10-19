"""
pixel_utils.py
이미지 전처리 / 픽셀아트 후처리 (GPU 불필요)
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# ── 팔레트 ────────────────────────────────────────────────────────────────────
AUTO_PALETTE = "🎨 자동 (이미지에서 K색 추출)"
KEEP_PALETTE = "🌈 원본 색 유지"

FIXED_PALETTES = {
    "🍬 파스텔 카와이": [
        "3D3047", "6D597A", "FFFFFF", "FFE4EC", "FFB3C6", "FF8FAB", "FB6F92",
        "C9E4DE", "A0D2C4", "BDE0FE", "A2D2FF", "CDB4DB", "FFF1B5", "FFD6A5",
    ],
    "🕹️ PICO-8 (16색)": [
        "000000", "1D2B53", "7E2553", "008751", "AB5236", "5F574F", "C2C3C7", "FFF1E8",
        "FF004D", "FFA300", "FFEC27", "00E436", "29ADFF", "83769C", "FF77A8", "FFCCAA",
    ],
    "🟩 게임보이 (4색)": ["0F380F", "306230", "8BAC0F", "9BBC0F"],
}

PALETTE_CHOICES = [AUTO_PALETTE, KEEP_PALETTE, *FIXED_PALETTES.keys()]


def _palette_image(hex_colors: list) -> Image.Image:
    flat = []
    for hx in hex_colors:
        flat += [int(hx[i:i + 2], 16) for i in (0, 2, 4)]
    flat += flat[:3] * (256 - len(hex_colors))
    pal = Image.new("P", (1, 1))
    pal.putpalette(flat)
    return pal


# ── 전처리 ────────────────────────────────────────────────────────────────────

def fit_image(img: Image.Image, long_side: int) -> Image.Image:
    """긴 변을 long_side에 맞추고 가로/세로를 64의 배수로 맞춤 (Diffusion 모델 요구사항)"""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = long_side / max(w, h)
    nw = max(64, round(w * s / 64) * 64)
    nh = max(64, round(h * s / 64) * 64)
    return img.resize((nw, nh), Image.LANCZOS)


def canny_image(img: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    edges = cv2.Canny(np.array(img), low, high)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


# ── 후처리: 진짜 픽셀아트로 만들기 ────────────────────────────────────────────

def pixelate(
    img: Image.Image,
    grid: int = 96,
    palette: str = AUTO_PALETTE,
    n_colors: int = 16,
    saturation: float = 1.2,
    contrast: float = 1.15,
    display_long_side: int = 768,
):
    """
    1) 작은 격자로 축소 (BOX = 칸마다 평균색)
    2) 채도/대비 부스트 (평균내며 흐려진 색을 또렷하고 귀엽게)
    3) 제한된 팔레트로 색 줄이기 (디더링 없음 → 깔끔한 면)
    4) Nearest Neighbor로 확대 (각진 픽셀 유지)

    반환: (화면 표시용 확대 이미지, 원본 픽셀 크기 스프라이트)
    """
    img = img.convert("RGB")
    w, h = img.size
    s = grid / max(w, h)
    sw, sh = max(1, round(w * s)), max(1, round(h * s))

    small = img.resize((sw, sh), Image.BOX)
    if saturation != 1.0:
        small = ImageEnhance.Color(small).enhance(saturation)
    if contrast != 1.0:
        small = ImageEnhance.Contrast(small).enhance(contrast)

    if palette == AUTO_PALETTE:
        small = small.quantize(
            colors=int(n_colors), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
        ).convert("RGB")
    elif palette in FIXED_PALETTES:
        small = small.quantize(
            palette=_palette_image(FIXED_PALETTES[palette]), dither=Image.Dither.NONE
        ).convert("RGB")

    factor = max(1, display_long_side // max(sw, sh))
    big = small.resize((sw * factor, sh * factor), Image.NEAREST)
    return big, small
