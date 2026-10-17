"""Cones as local anchors: in every frame, compare the cones we can see with where the cone map says they
should be. The median mismatch of the cones near the car = local registration error -> remove it from the
car's pixel position. (Distances still come from the ground plane + 75 m, NOT from cone spacing.)"""
import cv2, numpy as np
from scipy.signal import medfilt
import run2_setup as S


def detect_cones(f):
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    m = (((hsv[..., 0] < 12) | (hsv[..., 0] > 168)) & (hsv[..., 1] > 110) & (hsv[..., 2] > 80)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    return np.array([(x + w / 2, y + h) for x, y, w, h, a in st[1:] if 4 <= a <= 4000 and h >= 3 and w <= 2.5 * h])


def local_offsets(car_px, radius=120, match_px=5):
    """per frame: median (detected - predicted) pixel offset of cones within `radius` px of the car"""
    cone_map = np.load("work/run2_cone_map.npy")[:, :2] - S.OFF
    n = len(car_px); off = np.full((n, 2), np.nan)
    for i in range(n):
        if not np.all(np.isfinite(car_px[i])):
            continue
        det = detect_cones(cv2.imread(f"work/run2_jpg/{i:05d}.jpg"))
        if len(det) == 0:
            continue
        res = []
        for p in S.from_ref(i, cone_map):
            if np.hypot(*(p - car_px[i])) > radius:
                continue
            dd = np.hypot(*(det - p).T); j = np.argmin(dd)
            if dd[j] < match_px:
                res.append(det[j] - p)
        if len(res) >= 2:
            off[i] = np.median(res, axis=0)
    return off


def smooth_offsets(off, k=9):
    """median filter (kills single bad frames), fill gaps by interpolation, short moving average"""
    n = len(off); out = np.zeros_like(off); idx = np.arange(n)
    for c in range(2):
        ok = np.isfinite(off[:, c])
        v = np.interp(idx, idx[ok], off[ok, c])
        v = medfilt(v, k)
        out[:, c] = np.convolve(np.pad(v, 2, mode="edge"), np.ones(5) / 5, "valid")
    return out
