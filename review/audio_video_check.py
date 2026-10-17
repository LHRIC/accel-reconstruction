# Per-gear ratio of audio speed (RPM x gearing, Doppler removed) to video speed, for several tilts.
import numpy as np, sys, json
sys.path.insert(0, "."); sys.path.insert(0, "review")
import run2_setup as S
from tyre_track import lane_positions, axle_x, solve_scale
from smoother import rts_xva
d = np.load("work/run2_kin.npz"); tpx = d["tyre_px"] - d["cone_off_s"][:, None, :]
t = S.T_FRAME[:len(tpx)]; A = np.load("work/run2_rpm.npz"); ta, ra = A["t"], A["rpm_obs"]
C = 343.0; R = 0.198; PRIM = 2.111; FD = 2.84; GR = {2: 2.0, 3: 1.6667}
win = {2: (3.35, 4.35), 3: (4.75, 6.0)}            # heard-time windows, clean in-gear climbs
for vs in (-15, -20, -25, -30):
    k, origin, G = solve_scale(tpx, t, 13, v_shift=vs)
    xy, _ = lane_positions(tpx, k, v_shift=vs); x = axle_x(xy); y = np.nanmedian(xy[:, 1, 1]); ok = np.isfinite(x)
    X, _ = rts_xva(t[ok], x[ok], np.full(ok.sum(), 0.06), 40); tt = t[ok]
    out = []
    for g, (a, b) in win.items():
        te = np.linspace(a - 0.4, b, 500)
        dist = np.sqrt(np.interp(te, tt, X[:, 0]) ** 2 + y ** 2 + k ** 2); tob = te + dist / C
        m = (tob >= a) & (tob <= b); te, tob, dist = te[m], tob[m], dist[m]
        vv = np.interp(te, tt, X[:, 1]); vr = vv * np.interp(te, tt, X[:, 0]) / dist
        v_audio = np.interp(tob, ta, ra) * (C + vr) / C / (PRIM * FD * GR[g]) * 2 * np.pi * R / 60
        out.append(np.median(v_audio / vv))
    print(f"tilt {vs:+d}px: camera height {k:.2f} m | audio/video speed ratio  2nd gear {out[0]:.3f}  3rd gear {out[1]:.3f}  (3rd/2nd {out[1]/out[0]:.3f})")
