#!/usr/bin/env python3
"""Clean a finished occupancy map: remove thin free-space "spokes".

octomap clearing rays that slip through wall gaps carve thin free lines out to
max_range. They are only 1-3 cells wide, so a morphological OPENING of the free
mask (erode then dilate) erases them while leaving solid rooms, corridors and
the wider doorway fans intact. Optionally keep only the free region connected to
the map centre (drops fully-detached free islands).

Occupied cells are never removed. Removed free cells become 'unknown'.

Usage:
  python3 clean_occupancy_map.py in.pgm out_basename [--open-radius 2]
                                 [--keep-connected] [--free-thresh 250] [--occ-thresh 50]
Writes out_basename.pgm (+ copies/retargets out_basename.yaml if in.yaml exists).
Needs only numpy (+ scipy if present, for the connected-component pass).
"""
import argparse
import os
import numpy as np


def read_pgm(path):
    with open(path, 'rb') as f:
        assert f.readline().strip() == b'P5', 'not a binary PGM'
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        w, h = (int(x) for x in line.split())
        maxv = int(f.readline())
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w)
    return data


def write_pgm(path, data):
    h, w = data.shape
    with open(path, 'wb') as f:
        f.write(b'P5\n%d %d\n255\n' % (w, h))
        f.write(data.astype(np.uint8).tobytes())


def dilate(mask, r):
    m = mask
    for _ in range(max(0, r)):
        d = m.copy()
        d[1:, :] |= m[:-1, :]; d[:-1, :] |= m[1:, :]
        d[:, 1:] |= m[:, :-1]; d[:, :-1] |= m[:, 1:]
        d[1:, 1:] |= m[:-1, :-1]; d[:-1, :-1] |= m[1:, 1:]
        d[1:, :-1] |= m[:-1, 1:]; d[:-1, 1:] |= m[1:, :-1]
        m = d
    return m


def erode(mask, r):
    return ~dilate(~mask, r)


def opening(mask, r):
    return dilate(erode(mask, r), r) if r > 0 else mask


def keep_largest_component(free):
    try:
        from scipy import ndimage
    except Exception:
        print("  (scipy unavailable -> skipping connected-component pass)")
        return free
    lbl, n = ndimage.label(free)
    if n <= 1:
        return free
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    return lbl == sizes.argmax()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input_pgm')
    ap.add_argument('output_base')
    ap.add_argument('--open-radius', type=int, default=2,
                    help='erosion/dilation radius; removes free features thinner '
                         'than ~2*r cells (spokes). Default 2.')
    ap.add_argument('--keep-connected', action='store_true',
                    help='keep only the largest connected free region')
    ap.add_argument('--free-thresh', type=int, default=250)
    ap.add_argument('--occ-thresh', type=int, default=50)
    args = ap.parse_args()

    d = read_pgm(args.input_pgm)
    free = d >= args.free_thresh
    occ = d <= args.occ_thresh

    free_clean = opening(free, args.open_radius)
    if args.keep_connected:
        free_clean = keep_largest_component(free_clean)

    out = np.full(d.shape, 205, dtype=np.uint8)   # unknown
    out[free_clean] = 254                          # free
    out[occ] = 0                                    # occupied overrides

    removed = int(free.sum() - (free_clean & ~occ).sum())
    out_pgm = args.output_base + '.pgm'
    write_pgm(out_pgm, out)
    print(f"cleaned: {args.input_pgm} -> {out_pgm}")
    print(f"  free before {int(free.sum())}  after {int(free_clean.sum())}  "
          f"(removed {removed} spoke/island cells -> unknown)")

    in_yaml = os.path.splitext(args.input_pgm)[0] + '.yaml'
    if os.path.exists(in_yaml):
        with open(in_yaml) as f:
            y = f.read()
        base = os.path.basename(out_pgm)
        import re
        y = re.sub(r'image:\s*.*', f'image: {base}', y, count=1)
        with open(args.output_base + '.yaml', 'w') as f:
            f.write(y)
        print(f"  wrote {args.output_base}.yaml")


if __name__ == '__main__':
    main()
