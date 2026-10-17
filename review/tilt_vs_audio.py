# Within a gear, (audio speed)/(video speed) must be constant along the lane. A wrong tilt makes it drift.
import numpy as np, sys
sys.path.insert(0, "."); sys.path.insert(0, "review")
from calib3d import Ground
from smoother import rts_xva
Tm = np.load("work/run2_mosaicT.npy"); T = Tm[:9].reshape(3, 3); off = np.array([T[0, 2], T[1, 2]])
f0 = float(np.load("work/run2_fref.npy")[0]); V0 = np.load("work/run2_vps.npz")["V_lane"] - off
car = np.load("work/run2_car_raw.npy"); car = car[car[:, 1] >= 1.57]; t = car[:, 1]
A = np.load("work/run2_rpm.npz"); ta, ra = A["t"], A["rpm_obs"]
C, s = 343.0, 8.0
for vs in [-5, -20, -35]:
    G = Ground(f0, V0 + [0, vs], 5.1, h=1.0); P = G.to_lane(car[:, 2:4] - off)
    X, _ = rts_xva(t, P[:, 0], np.full(len(t), 0.012), 0.02); xu, vu = X[:, 0], X[:, 1]; yu = np.median(P[:, 1])
    out = []
    for (a, b) in [(3.35, 4.40), (4.75, 6.00)]:
        te = np.linspace(a - 0.3, b, 400); d = s * np.sqrt(np.interp(te, t, xu) ** 2 + yu ** 2 + 1); tob = te + d / C
        m = (tob >= a) & (tob <= b); te, tob, d = te[m], tob[m], d[m]
        vr = s * np.interp(te, t, vu) * s * np.interp(te, t, xu) / d
        rpm_true = np.interp(tob, ta, ra) * (C + vr) / C
        ratio = rpm_true / np.interp(te, t, vu)                       # RPM per unit video speed (should be flat)
        slope = np.polyfit(np.interp(te, t, xu) * s, ratio / ratio.mean(), 1)[0] * 100   # % change per 100 m... per m*100
        out.append(f"{slope*100:+.1f}%/100m")
    print(f"V_shift {vs:4d}: drift of RPM/v_video along the lane  2nd gear {out[0]}   3rd gear {out[1]}")
