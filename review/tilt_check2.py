import numpy as np, sys
sys.path.insert(0, "."); sys.path.insert(0, "review")
from calib3d import Ground
from smoother import rts_xva
H0 = np.load("work/run2_kf.npz")["H_0i"]; Tm = np.load("work/run2_mosaicT.npy"); T = Tm[:9].reshape(3, 3); ref = int(Tm[9])
off = np.array([T[0, 2], T[1, 2]]); f0 = float(np.load("work/run2_fref.npy")[0]); V0 = np.load("work/run2_vps.npz")["V_lane"] - off
dash = np.load("work/run2_dash_ref_px.npz")
def to_ref(i, uv):
    M = np.linalg.inv(H0[ref]) @ H0[i]; uv = np.atleast_2d(uv); q = M @ np.c_[uv, np.ones(len(uv))].T; return (q[:2] / q[2]).T
SL = to_ref(0, [(360, 106), (557, 132)]); FL = to_ref(300, [(275, 116), (420, 129)])
car = np.load("work/run2_car_raw.npy"); car = car[car[:, 1] >= 1.57]; t = car[:, 1]; car_ref = car[:, 2:4] - off
def x_at(p, y): a = (y - p[0, 1]) / (p[1, 1] - p[0, 1]); return p[0, 0] + a * (p[1, 0] - p[0, 0])
for vs in [0, -20, -35]:
    G = Ground(f0, V0 + [0, vs], 5.1, h=1.0)
    P = G.to_lane(car_ref); yc = np.median(P[:, 1])
    xs_, xf_ = x_at(G.to_lane(SL), yc), x_at(G.to_lane(FL), yc)
    s = 75 / (xf_ - xs_)                                    # H1: white lines are 75 m apart
    ds = G.to_lane(dash["start"])[:-1, 0] * s; de = G.to_lane(dash["end"])[:-1, 0] * s; dc = (ds + de) / 2
    x = s * (P[:, 0] - xs_); X, S = rts_xva(t, x, np.full(len(t), 0.1), 20)
    t_cross = lambda xx: np.interp(xx, X[:, 0], t)
    print(f"Vshift {vs:4d}: cam height {s:5.2f} m | dash centre gaps {np.diff(dc).round(2)} | dash len {(de-ds).round(2)}")
    print(f"            car crosses white start line at t={t_cross(0):.3f}s (v={np.interp(t_cross(0), t, X[:,1]):.1f} m/s), "
          f"white finish line at t={t_cross(75):.3f}s -> elapsed {t_cross(75)-t_cross(0):.3f}s (official 4.837)")
