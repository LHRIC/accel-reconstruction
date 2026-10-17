# Independent scale: video speed (unscaled) vs audio speed (RPM x gearing x tyre radius, Doppler + delay corrected)
import numpy as np, sys
sys.path.insert(0, "."); sys.path.insert(0, "review")
from calib3d import Ground
from smoother import rts_xva
Tm = np.load("work/run2_mosaicT.npy"); T = Tm[:9].reshape(3, 3); off = np.array([T[0, 2], T[1, 2]])
f0 = float(np.load("work/run2_fref.npy")[0]); V0 = np.load("work/run2_vps.npz")["V_lane"] - off
G = Ground(f0, V0 + [0, -20], 5.1, h=1.0)
car = np.load("work/run2_car_raw.npy"); car = car[car[:, 1] >= 1.57]; t = car[:, 1]
P = G.to_lane(car[:, 2:4] - off)                                   # unscaled lane coords (camera height = 1)
X, S = rts_xva(t, P[:, 0], np.full(len(t), 0.012), 0.02)          # smooth unscaled x (units of camera height)
xu, vu = X[:, 0], X[:, 1]; yu = np.median(P[:, 1])
A = np.load("work/run2_rpm.npz"); ta, ra = A["t"], A["rpm_obs"]
C = 343.0; r_tyre = 0.198; prim, fd = 2.111, 2.84; gears = {1: 2.75, 2: 2.0, 3: 1.6667}
seg = {1: (2.10, 2.65), 2: (3.35, 4.40), 3: (4.75, 6.00)}           # audio-time windows, clean in-gear climbs
for s in [8.08, 7.3]:                                              # try two camera heights (scale) for the geometry terms
    print(f"--- geometry with camera height {s} m")
    for g, (a, b) in seg.items():
        te = np.linspace(a - 0.3, b, 300)                          # candidate emission times
        d = s * np.sqrt(np.interp(te, t, xu) ** 2 + yu ** 2 + 1.0)  # camera->car distance [m]
        tob = te + d / C                                            # when that sound reaches the phone
        m = (tob >= a) & (tob <= b); te, tob, d = te[m], tob[m], d[m]
        vr = s * np.interp(te, t, vu) * (s * np.interp(te, t, xu)) / d   # receding speed along line of sight
        rpm_true = np.interp(tob, ta, ra) * (C + vr) / C
        v_audio = rpm_true / (prim * fd * gears[g]) * 2 * np.pi * r_tyre / 60
        k = np.median(v_audio / np.interp(te, t, vu))              # scale = metres per unscaled unit
        print(f"  gear {g}: audio speed {v_audio.min():5.1f}-{v_audio.max():5.1f} m/s, implied camera height (scale) {k:5.2f} m")
