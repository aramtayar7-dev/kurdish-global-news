"""
Kurdish Global News Bot — Default Fallback Background Generator

Generates 11 high-contrast, broadcast-quality 1280x720 fallback backgrounds
for articles without images. Each background reflects its region or topic
with distinct dark palettes, subtle architectural/tactical silhouettes,
and zero text, zero watermarks, and zero national flags.
"""

from __future__ import annotations

import math
from pathlib import Path
from PIL import Image, ImageDraw

CARD_WIDTH = 1280
CARD_HEIGHT = 720


def _draw_vertical_gradient(
    draw: ImageDraw.ImageDraw,
    top_color: tuple[int, int, int],
    bottom_color: tuple[int, int, int],
) -> None:
    """Draw a smooth vertical color gradient across the entire canvas."""
    r1, g1, b1 = top_color
    r2, g2, b2 = bottom_color
    for y in range(CARD_HEIGHT):
        t = y / CARD_HEIGHT
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        draw.line([(0, y), (CARD_WIDTH, y)], fill=(r, g, b, 255))


def _apply_common_overlays(img: Image.Image) -> Image.Image:
    """
    Apply bottom 55% headline-legibility gradient, top vignette,
    and crisp crimson top border accent.
    """
    overlay = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Dark gradient over bottom 55% (starts at y = 324)
    start_y = int(CARD_HEIGHT * 0.45)
    span = CARD_HEIGHT - start_y
    for y in range(start_y, CARD_HEIGHT):
        prog = (y - start_y) / max(span, 1)
        alpha = int(240 * (prog ** 1.3))
        draw.line([(0, y), (CARD_WIDTH, y)], fill=(8, 12, 22, alpha))

    # Top-right vignette to guarantee KGN logo / badge contrast
    for y in range(0, 140):
        prog = 1.0 - (y / 140.0)
        alpha = int(120 * prog)
        draw.line([(0, y), (CARD_WIDTH, y)], fill=(8, 12, 22, alpha))

    # Crimson top broadcast line
    draw.rectangle([(0, 0), (CARD_WIDTH, 5)], fill=(220, 38, 38, 255))

    return Image.alpha_composite(img, overlay)


# ═══════════════════════════════════════════════════
# 1. Kurdistan Bashur (Southern Kurdistan)
# ═══════════════════════════════════════════════════
def generate_kurdistan_bashur() -> Image.Image:
    """Deep slate & midnight navy with Zagros mountain silhouettes & golden dawn horizon."""
    base = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (10, 15, 29, 255))
    draw = ImageDraw.Draw(base)
    _draw_vertical_gradient(draw, (11, 17, 30), (18, 28, 48))

    overlay = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # Subtle golden sunrise glow on mid-left horizon
    for r in range(220, 0, -5):
        alpha = int(35 * (1.0 - (r / 220.0)))
        odraw.ellipse([(200 - r, 360 - r // 2), (200 + r, 360 + r // 2)], fill=(217, 119, 6, alpha))

    # Distant mountain ridge (slate blue)
    dist_ridge = [
        (0, 380), (140, 310), (280, 350), (450, 280), (600, 330),
        (750, 290), (900, 340), (1080, 270), (1280, 330),
        (1280, CARD_HEIGHT), (0, CARD_HEIGHT)
    ]
    odraw.polygon(dist_ridge, fill=(24, 35, 56, 180))

    # Foreground mountain ridge (deep charcoal navy)
    fore_ridge = [
        (0, 440), (180, 370), (340, 420), (520, 350), (700, 410),
        (880, 360), (1040, 400), (1200, 350), (1280, 380),
        (1280, CARD_HEIGHT), (0, CARD_HEIGHT)
    ]
    odraw.polygon(fore_ridge, fill=(15, 23, 38, 220))

    # Subtle crimson angled broadcast facet on far left
    odraw.polygon([(0, 0), (280, 0), (80, CARD_HEIGHT), (0, CARD_HEIGHT)], fill=(185, 28, 28, 30))

    combined = Image.alpha_composite(base, overlay)
    return _apply_common_overlays(combined)


# ═══════════════════════════════════════════════════
# 2. Iraq Baghdad (Federal Iraq)
# ═══════════════════════════════════════════════════
def generate_iraq_baghdad() -> Image.Image:
    """Dark obsidian & midnight bronze with Mesopotamian arches & subtle gold lattice."""
    base = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (12, 14, 20, 255))
    draw = ImageDraw.Draw(base)
    _draw_vertical_gradient(draw, (12, 15, 22), (24, 27, 36))

    overlay = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # Geometric arches across middle ground
    arch_w = 140
    for x in range(40, CARD_WIDTH - 40, arch_w + 30):
        top_y = 230
        odraw.arc([(x, top_y), (x + arch_w, top_y + 160)], start=180, end=360, fill=(180, 130, 45, 45), width=2)
        odraw.line([(x, top_y + 80), (x, 480)], fill=(180, 130, 45, 30), width=1)
        odraw.line([(x + arch_w, top_y + 80), (x + arch_w, 480)], fill=(180, 130, 45, 30), width=1)

    # Subtle warm amber horizon line
    odraw.line([(0, 360), (CARD_WIDTH, 360)], fill=(180, 110, 30, 60), width=2)

    # Angular broadcast facet
    odraw.polygon([(0, 0), (360, 0), (120, CARD_HEIGHT), (0, CARD_HEIGHT)], fill=(180, 83, 9, 25))

    combined = Image.alpha_composite(base, overlay)
    return _apply_common_overlays(combined)


# ═══════════════════════════════════════════════════
# 3. Iran Tehran
# ═══════════════════════════════════════════════════
def generate_iran_tehran() -> Image.Image:
    """Deep midnight teal & charcoal with Persian geometric lattice & turquoise tactical accents."""
    base = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (8, 18, 24, 255))
    draw = ImageDraw.Draw(base)
    _draw_vertical_gradient(draw, (7, 18, 24), (16, 34, 44))

    overlay = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # Persian isometric diamond lattice
    spacing = 70
    for x in range(-CARD_HEIGHT, CARD_WIDTH + CARD_HEIGHT, spacing):
        odraw.line([(x, 0), (x + CARD_HEIGHT, CARD_HEIGHT)], fill=(13, 148, 136, 16), width=1)
        odraw.line([(x, 0), (x - CARD_HEIGHT, CARD_HEIGHT)], fill=(13, 148, 136, 16), width=1)

    # Central tactical horizon glow
    for r in range(180, 0, -6):
        alpha = int(28 * (1.0 - (r / 180.0)))
        odraw.ellipse([(640 - r, 320 - r // 2), (640 + r, 320 + r // 2)], fill=(6, 182, 212, alpha))

    # Angular turquoise facet
    odraw.polygon([(0, 0), (320, 0), (90, CARD_HEIGHT), (0, CARD_HEIGHT)], fill=(15, 118, 110, 30))

    combined = Image.alpha_composite(base, overlay)
    return _apply_common_overlays(combined)


# ═══════════════════════════════════════════════════
# 4. Israel Levant (Israel, Gaza, Lebanon)
# ═══════════════════════════════════════════════════
def generate_israel_levant() -> Image.Image:
    """Deep Mediterranean navy & midnight cobalt with tactical horizon waves & silver broadcast facets."""
    base = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (7, 15, 30, 255))
    draw = ImageDraw.Draw(base)
    _draw_vertical_gradient(draw, (8, 16, 32), (18, 32, 60))

    overlay = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # Diagonal tactical broadcast bands
    odraw.polygon([(0, 0), (480, 0), (220, CARD_HEIGHT), (0, CARD_HEIGHT)], fill=(30, 58, 138, 50))
    odraw.polygon([(CARD_WIDTH - 420, 0), (CARD_WIDTH, 0), (CARD_WIDTH, 340)], fill=(37, 99, 235, 25))

    # Wave-like horizontal coastal lines
    for y_wave in range(260, 420, 30):
        points = []
        for x in range(0, CARD_WIDTH + 40, 40):
            y_offset = int(math.sin(x * 0.015) * 8)
            points.append((x, y_wave + y_offset))
        odraw.line(points, fill=(96, 165, 250, 25), width=1)

    combined = Image.alpha_composite(base, overlay)
    return _apply_common_overlays(combined)


# ═══════════════════════════════════════════════════
# 5. Syria Damascus (Syria & Rojava)
# ═══════════════════════════════════════════════════
def generate_syria_damascus() -> Image.Image:
    """Dark desert dusk & umber slate with Levantine colonnade silhouettes & terracotta accents."""
    base = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (18, 14, 12, 255))
    draw = ImageDraw.Draw(base)
    _draw_vertical_gradient(draw, (18, 14, 12), (36, 26, 22))

    overlay = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # Warm desert sunset glow
    for r in range(200, 0, -6):
        alpha = int(32 * (1.0 - (r / 200.0)))
        odraw.ellipse([(400 - r, 330 - r // 2), (400 + r, 330 + r // 2)], fill=(194, 65, 12, alpha))

    # Classical colonnade pillars across mid-ground
    col_w = 16
    spacing = 65
    for x in range(120, 1160, spacing):
        odraw.rectangle([(x, 240), (x + col_w, 460)], fill=(44, 32, 26, 120))
        # Architrave beam
        odraw.rectangle([(x - 8, 232), (x + col_w + 8, 242)], fill=(56, 40, 32, 140))
    odraw.rectangle([(100, 224), (1180, 234)], fill=(65, 46, 36, 150))

    # Deep terracotta angled facet
    odraw.polygon([(0, 0), (320, 0), (100, CARD_HEIGHT), (0, CARD_HEIGHT)], fill=(154, 52, 18, 30))

    combined = Image.alpha_composite(base, overlay)
    return _apply_common_overlays(combined)


# ═══════════════════════════════════════════════════
# 6. Turkey Ankara (Turkey & Bakur)
# ═══════════════════════════════════════════════════
def generate_turkey_ankara() -> Image.Image:
    """Deep crimson dusk & dark slate with geometric bridge cables & polygonal broadcast facets."""
    base = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (20, 10, 13, 255))
    draw = ImageDraw.Draw(base)
    _draw_vertical_gradient(draw, (20, 10, 13), (38, 16, 22))

    overlay = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # Crimson angled broadcast facets
    odraw.polygon([(0, 0), (460, 0), (180, CARD_HEIGHT), (0, CARD_HEIGHT)], fill=(153, 27, 27, 45))
    odraw.polygon([(CARD_WIDTH - 380, 0), (CARD_WIDTH, 0), (CARD_WIDTH, 280)], fill=(220, 38, 38, 20))

    # Suspension bridge / radiating cables silhouette from upper center
    center_x = 420
    center_y = 120
    for end_x in range(60, 900, 45):
        odraw.line([(center_x, center_y), (end_x, 420)], fill=(225, 29, 72, 22), width=1)
    odraw.line([(40, 420), (920, 420)], fill=(225, 29, 72, 45), width=2)

    combined = Image.alpha_composite(base, overlay)
    return _apply_common_overlays(combined)


# ═══════════════════════════════════════════════════
# 7. USA Washington
# ═══════════════════════════════════════════════════
def generate_usa_washington() -> Image.Image:
    """Midnight navy & dark titanium slate with neoclassical dome/columns silhouette."""
    base = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (8, 13, 26, 255))
    draw = ImageDraw.Draw(base)
    _draw_vertical_gradient(draw, (9, 14, 28), (20, 30, 52))

    overlay = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # Neoclassical Capitol dome silhouette on lower-left
    dome_cx = 260
    dome_base_y = 440
    # Pedestal & drum
    odraw.rectangle([(dome_cx - 90, dome_base_y - 80), (dome_cx + 90, dome_base_y)], fill=(20, 28, 48, 160))
    # Dome semi-ellipse
    odraw.pieslice([(dome_cx - 75, dome_base_y - 180), (dome_cx + 75, dome_base_y - 40)], start=180, end=360, fill=(24, 34, 58, 170))
    # Cupola & lantern
    odraw.rectangle([(dome_cx - 15, dome_base_y - 215), (dome_cx + 15, dome_base_y - 180)], fill=(28, 40, 68, 180))
    odraw.line([(dome_cx, dome_base_y - 235), (dome_cx, dome_base_y - 215)], fill=(148, 163, 184, 180), width=2)

    # Portico columns flanking
    for cx in range(dome_cx - 160, dome_cx - 90, 22):
        odraw.rectangle([(cx, dome_base_y - 60), (cx + 10, dome_base_y)], fill=(18, 25, 42, 140))
    for cx in range(dome_cx + 100, dome_cx + 170, 22):
        odraw.rectangle([(cx, dome_base_y - 60), (cx + 10, dome_base_y)], fill=(18, 25, 42, 140))

    # Crisp vertical blue & crimson accent pinstripes
    odraw.line([(70, 0), (70, CARD_HEIGHT)], fill=(37, 99, 235, 50), width=2)
    odraw.line([(78, 0), (78, CARD_HEIGHT)], fill=(220, 38, 38, 40), width=2)

    combined = Image.alpha_composite(base, overlay)
    return _apply_common_overlays(combined)


# ═══════════════════════════════════════════════════
# 8. Cybersecurity
# ═══════════════════════════════════════════════════
def generate_cybersecurity() -> Image.Image:
    """Dark cyber obsidian with digital circuit nodes, cyber matrix grid, & glowing cyan traces."""
    base = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (5, 11, 20, 255))
    draw = ImageDraw.Draw(base)
    _draw_vertical_gradient(draw, (5, 11, 20), (10, 25, 45))

    overlay = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # Perspective cyber grid
    grid_spacing = 60
    for x in range(0, CARD_WIDTH, grid_spacing):
        odraw.line([(x, 100), (x, CARD_HEIGHT)], fill=(6, 182, 212, 14), width=1)
    for y in range(120, CARD_HEIGHT, 50):
        odraw.line([(0, y), (CARD_WIDTH, y)], fill=(6, 182, 212, 12), width=1)

    # Tactical circuit node network
    nodes = [
        (180, 220), (280, 180), (420, 240), (560, 190),
        (720, 250), (860, 200), (320, 310), (510, 340), (680, 320)
    ]
    for i, (nx, ny) in enumerate(nodes):
        # Connected traces
        if i + 1 < len(nodes):
            next_x, next_y = nodes[i + 1]
            odraw.line([(nx, ny), (next_x, next_y)], fill=(14, 165, 233, 40), width=1)
        # Node glowing halo & core
        odraw.ellipse([(nx - 8, ny - 8), (nx + 8, ny + 8)], fill=(6, 182, 212, 50))
        odraw.ellipse([(nx - 3, ny - 3), (nx + 3, ny + 3)], fill=(255, 255, 255, 180))

    combined = Image.alpha_composite(base, overlay)
    return _apply_common_overlays(combined)


# ═══════════════════════════════════════════════════
# 9. Economy Energy
# ═══════════════════════════════════════════════════
def generate_economy_energy() -> Image.Image:
    """Dark financial emerald & obsidian with candlestick chart silhouette & ascending gold trend line."""
    base = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (6, 20, 14, 255))
    draw = ImageDraw.Draw(base)
    _draw_vertical_gradient(draw, (6, 19, 14), (12, 38, 27))

    overlay = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # Candlestick chart silhouettes across mid-ground
    candles = [
        (120, 380, 430, 350, 450, True),
        (180, 340, 400, 320, 415, True),
        (240, 360, 330, 310, 380, False),
        (300, 320, 370, 290, 390, True),
        (360, 290, 340, 270, 360, True),
        (420, 310, 280, 260, 330, False),
        (480, 270, 310, 250, 330, True),
        (540, 240, 290, 220, 305, True),
        (600, 260, 230, 210, 280, False),
        (660, 220, 270, 190, 285, True),
        (720, 200, 240, 180, 260, True),
        (780, 230, 190, 170, 245, False),
        (840, 180, 220, 160, 235, True),
    ]
    for cx, c_top, c_bot, wick_top, wick_bot, is_green in candles:
        wick_color = (16, 185, 129, 65) if is_green else (239, 68, 68, 50)
        body_color = (16, 185, 129, 85) if is_green else (239, 68, 68, 70)
        odraw.line([(cx, wick_top), (cx, wick_bot)], fill=wick_color, width=2)
        odraw.rectangle([(cx - 10, min(c_top, c_bot)), (cx + 10, max(c_top, c_bot))], fill=body_color)

    # Ascending golden trendline
    trend_pts = [(cx, (c_top + c_bot) // 2) for cx, c_top, c_bot, _, _, _ in candles]
    odraw.line(trend_pts, fill=(245, 158, 11, 100), width=3)

    combined = Image.alpha_composite(base, overlay)
    return _apply_common_overlays(combined)


# ═══════════════════════════════════════════════════
# 10. Military Conflict
# ═══════════════════════════════════════════════════
def generate_military_conflict() -> Image.Image:
    """Dark tactical slate & ember charcoal with concentric radar rings & ember crimson facets."""
    base = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (20, 8, 8, 255))
    draw = ImageDraw.Draw(base)
    _draw_vertical_gradient(draw, (20, 8, 8), (36, 14, 14))

    overlay = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # Concentric tactical radar sweep rings on mid-left
    rcx, rcy = 380, 280
    for radius in (60, 120, 190, 270, 360):
        odraw.ellipse([(rcx - radius, rcy - radius), (rcx + radius, rcy + radius)], outline=(239, 68, 68, 30), width=1)
    # Crosshair axes
    odraw.line([(rcx - 380, rcy), (rcx + 380, rcy)], fill=(239, 68, 68, 35), width=1)
    odraw.line([(rcx, rcy - 280), (rcx, rcy + 280)], fill=(239, 68, 68, 35), width=1)

    # Angular stealth facet overlays
    odraw.polygon([(0, 0), (480, 0), (200, CARD_HEIGHT), (0, CARD_HEIGHT)], fill=(153, 27, 27, 45))
    odraw.polygon([(CARD_WIDTH - 420, 0), (CARD_WIDTH, 0), (CARD_WIDTH, 340)], fill=(185, 28, 28, 25))

    combined = Image.alpha_composite(base, overlay)
    return _apply_common_overlays(combined)


# ═══════════════════════════════════════════════════
# 11. Global Diplomacy
# ═══════════════════════════════════════════════════
def generate_global_diplomacy() -> Image.Image:
    """Deep diplomatic navy & sapphire with orthographic globe wireframe & broadcast grid."""
    base = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (7, 13, 30, 255))
    draw = ImageDraw.Draw(base)
    _draw_vertical_gradient(draw, (7, 13, 30), (14, 25, 52))

    overlay = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # Orthographic globe wireframe on mid-left
    gcx, gcy, gr = 360, 280, 220
    # Outer circle
    odraw.ellipse([(gcx - gr, gcy - gr), (gcx + gr, gcy + gr)], outline=(56, 189, 248, 50), width=2)
    # Latitude lines
    for lat_scale in (0.35, 0.7, 0.92):
        lat_h = int(gr * lat_scale)
        odraw.ellipse([(gcx - gr, gcy - lat_h), (gcx + gr, gcy + lat_h)], outline=(56, 189, 248, 22), width=1)
    # Longitude ellipses
    for lon_scale in (0.35, 0.7):
        lon_w = int(gr * lon_scale)
        odraw.ellipse([(gcx - lon_w, gcy - gr), (gcx + lon_w, gcy + gr)], outline=(56, 189, 248, 25), width=1)
    # Equator & Prime Meridian
    odraw.line([(gcx - gr, gcy), (gcx + gr, gcy)], fill=(56, 189, 248, 45), width=1)
    odraw.line([(gcx, gcy - gr), (gcx, gcy + gr)], fill=(56, 189, 248, 45), width=1)

    # Clean diagonal diplomatic facets
    odraw.polygon([(0, 0), (440, 0), (180, CARD_HEIGHT), (0, CARD_HEIGHT)], fill=(30, 58, 138, 40))
    odraw.polygon([(CARD_WIDTH - 360, 0), (CARD_WIDTH, 0), (CARD_WIDTH, 260)], fill=(2, 132, 199, 20))

    combined = Image.alpha_composite(base, overlay)
    return _apply_common_overlays(combined)


# ═══════════════════════════════════════════════════
# Theme Dispatcher
# ═══════════════════════════════════════════════════
THEME_GENERATORS = {
    "kurdistan_bashur": generate_kurdistan_bashur,
    "iraq_baghdad": generate_iraq_baghdad,
    "iran_tehran": generate_iran_tehran,
    "israel_levant": generate_israel_levant,
    "syria_damascus": generate_syria_damascus,
    "turkey_ankara": generate_turkey_ankara,
    "usa_washington": generate_usa_washington,
    "cybersecurity": generate_cybersecurity,
    "economy_energy": generate_economy_energy,
    "military_conflict": generate_military_conflict,
    "global_diplomacy": generate_global_diplomacy,
}


def generate_all_backgrounds(output_dir: Path) -> dict[str, Path]:
    """Generate all 11 canonical fallback backgrounds as 1280x720 PNG files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}

    for theme_name, gen_func in THEME_GENERATORS.items():
        out_path = output_dir / f"{theme_name}.png"
        img = gen_func()
        # Ensure RGB 1280x720 output
        rgb_img = img.convert("RGB")
        rgb_img.save(out_path, format="PNG", optimize=True)
        generated[theme_name] = out_path

    return generated


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parent
    target_dir = repo_root / "assets" / "backgrounds"

    print(f"Generating 11 canonical fallback backgrounds in: {target_dir}")
    results = generate_all_backgrounds(target_dir)
    for theme, path in results.items():
        print(f"  ✓ {theme}: {path.name} ({path.stat().st_size // 1024} KB)")
    print("Done! All 11 default backgrounds successfully created.")
