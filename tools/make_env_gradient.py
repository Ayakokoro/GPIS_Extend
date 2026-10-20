#!/usr/bin/env python3
"""Bake the analytic +Z sky gradient to Tungsten's infinite_sphere lat-long map.

The RGB function matches primitive_macrofacet/make_env_gradient.py:
    t = (direction.z + 1) / 2
    RGB = (0.15 + 0.85*t, 0.20 + 0.70*t, 0.35 + 0.55*t)

Tungsten maps a world direction to u = atan2(z, x)/(2*pi) + 0.5 and
v = acos(-y)/pi. Its BitmapTexture places v=1 at the image's top row.
Consequently this unrotated map has +Y at the top and +Z on the equator at
u=0.75. The gradient still follows world +Z, as in the pbrt environment.

The output is a linear RGB PFM, which Tungsten reads directly as an emission
texture. PFM stores its bottom row first; Tungsten's PFM loader reverses that
order when loading it.

Example, from the Tungsten project root:
    python tools/make_env_gradient.py

Then set the infinite_sphere emission in a scene to
"gradient_latlong.pfm" (relative to that scene's directory).
"""

import argparse
import math
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "gpis-scenes" / "shaderball_ply" / "gradient_latlong.pfm"
)


def bake(output: Path, width: int, height: int) -> None:
    # The map's horizontal coordinate spans phi in [-pi, pi].
    u = (np.arange(width, dtype=np.float64) + 0.5) / width
    sin_phi = np.sin((u - 0.5) * (2.0 * math.pi))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(f"PF\n{width} {height}\n-1.0\n".encode("ascii"))

        # y=0 is the image's top row. Tungsten's v=1 is at the top, while
        # PFM requires rows to be written from bottom to top.
        for y in range(height - 1, -1, -1):
            v = 1.0 - (y + 0.5) / height
            sin_theta = math.sin(math.pi * v)
            direction_z = sin_phi * sin_theta
            t = 0.5 * (direction_z + 1.0)

            row = np.empty((width, 3), dtype="<f4")
            row[:, 0] = 0.15 + 0.85 * t
            row[:, 1] = 0.20 + 0.70 * t
            row[:, 2] = 0.35 + 0.55 * t
            stream.write(row.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=4096,
                        help="map width; height defaults to half this value")
    parser.add_argument("--height", type=int,
                        help="map height (must be half the width)")
    parser.add_argument("--outfile", type=Path, default=DEFAULT_OUTPUT,
                        help="output .pfm file; directly usable by Tungsten")
    args = parser.parse_args()

    height = args.height if args.height is not None else args.width // 2
    if args.width < 2 or height < 1 or args.width != 2 * height:
        parser.error("a lat-long map requires positive 2:1 width and height")
    if args.outfile.suffix.lower() != ".pfm":
        parser.error("output filename must end in .pfm")

    bake(args.outfile, args.width, height)
    print(f"Wrote {args.width}x{height} Tungsten lat-long map: {args.outfile}")


if __name__ == "__main__":
    main()
