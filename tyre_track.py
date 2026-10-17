"""SAM2 masks -> rear tyre centre pixel per frame -> lane position (metres).
Scale: 75 m from the start line (front wheels staged 0.3 m behind it) to the painted finish line."""
import numpy as np
import run2_setup as S

def load_masks():
    d = np.load("work/run2_masks.npz"); n, k, h, w = d["shape"]
    return np.unpackbits(d["packed"], axis=-1)[..., :w].astype(bool).reshape(n, k, h, w)

def tyre_pixels(masks, min_area=3):
    """centroid of each rear-tyre mask (obj 2 = far-side tyre, obj 3 = near-side tyre); NaN if missing"""
    n = len(masks); out = np.full((n, 2, 2), np.nan); area = np.zeros((n, 2))
    for i in range(n):
        for j, obj in enumerate((1, 2)):
            ys, xs = np.nonzero(masks[i, obj])
            area[i, j] = len(xs)
            if len(xs) >= min_area: out[i, j] = (xs.mean(), ys.mean())
    return out, area

def lane_positions(tyre_px, k_scale, v_shift=S.V_SHIFT):
    """tyre centre pixels -> lane (x, y) in metres for camera height k_scale"""
    G = S.ground(h=k_scale, v_shift=v_shift)
    n = len(tyre_px); xy = np.full((n, 2, 2), np.nan)
    for i in range(n):
        for j in range(2):
            if np.isfinite(tyre_px[i, j, 0]):
                xy[i, j] = G.to_lane(S.to_ref(i, tyre_px[i, j]), z=S.TYRE_CENTRE_Z)[0]
    return xy, G

def axle_x(xy):
    """rear axle position = NEAR-side ("visible") rear tyre; where it's hidden, far-side tyre + local far-near offset"""
    n = len(xy); x_far, x_near = xy[:, 0, 0], xy[:, 1, 0]
    both = np.isfinite(x_far) & np.isfinite(x_near)
    off = np.interp(np.arange(n), np.flatnonzero(both), np.convolve((x_far - x_near)[both], np.ones(15) / 15, "same"))
    return np.where(np.isfinite(x_near), x_near, x_far - off)


def solve_scale(tyre_px, t, launch_frame, v_shift=S.V_SHIFT):
    """camera height k so that (start line = staged front wheels + 0.3 m) -> painted finish line = 75 m"""
    k = 8.0
    for _ in range(4):                                       # tyre-height term depends on k, so iterate
        xy, G = lane_positions(tyre_px, k, v_shift)
        x = axle_x(xy); y = np.nanmedian(xy[:, 1, 1])
        x_staged = np.nanmedian(x[:launch_frame])
        x_finish = S.line_x_at(G, S.FINISH_LINE, y)
        start_line = x_staged + S.WHEELBASE + 0.3
        k = k * S.COURSE / (x_finish - start_line)
    xy, G = lane_positions(tyre_px, k, v_shift)
    x = axle_x(xy)
    x_staged = np.nanmedian(x[:launch_frame])
    origin = x_staged + S.WHEELBASE + 0.3                    # lane x of the start line
    return k, origin, G
