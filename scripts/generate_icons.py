#!/usr/bin/env python3
"""Generate favicon and app icons from aero_trade_logo.png."""

import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Install Pillow: pip install Pillow")
    sys.exit(1)

SRC = Path("aero_trade_logo.png")
OUT = Path("assets/branding")
UI_PUBLIC = Path("ui/public")

SIZES = {
    "aero_trade_logo.png": None,
    "favicon-16.png": 16,
    "favicon-32.png": 32,
    "favicon-48.png": 48,
    "apple-touch-icon.png": 180,
    "icon-192.png": 192,
    "icon-512.png": 512,
}


def main():
    if not SRC.exists():
        print(f"Logo not found: {SRC}")
        sys.exit(1)

    img = Image.open(SRC).convert("RGBA")
    w, h = img.size
    print(f"Source logo: {w}x{h}px")

    if w < 192:
        print("WARNING: Logo resolution may be too low for large desktop icons.")
        print("Consider providing a higher-resolution export for best results.")

    for d in (OUT, UI_PUBLIC):
        d.mkdir(parents=True, exist_ok=True)

    for name, size in SIZES.items():
        dest = OUT / name
        if size is None:
            img.save(dest)
        else:
            resized = img.resize((size, size), Image.LANCZOS)
            resized.save(dest)
        print(f"  -> {dest}")

    # Generate .ico for Windows
    ico_path = OUT / "favicon.ico"
    icons = [img.resize((s, s), Image.LANCZOS) for s in [16, 32, 48]]
    icons[0].save(ico_path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48)])
    print(f"  -> {ico_path}")

    # Copy to UI public
    for f in OUT.glob("*"):
        (UI_PUBLIC / f.name).write_bytes(f.read_bytes())
    print("Icons copied to ui/public/")


if __name__ == "__main__":
    main()
