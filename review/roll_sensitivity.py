# Does the plane roll matter once each version is scaled to start->finish = 75 m?
import numpy as np, sys
sys.path.insert(0, "."); sys.path.insert(0, "review")
from calib3d import Ground
from smoother import rts_xva
H0 = np.load("work/run2_kf.npz")["H_0i"]; Tm = np.load("work/run2_mosaicT.npy"); T = Tm[:9].reshape(3, 3); ref = int(Tm[9])
off = np.array([T[0, 2], T[1, 2]]); f_ref = float(np.load("work/run2_fref.npy")[0])
V_lane = np.load("work/run2_vps.npz")["V_lane"] - off
def to_ref(i, uv):
    M = np.linalg.inv(H0[ref]) @ H0[i]; uv = np.atleast_2d(uv); q = M @ np.c_[uv, np.ones(len(uv))].T
    return (q[:2] / q[2]).T
start_line = to_ref(0, [(360, 106), (557, 132)]); finish_line = to_ref(300, [(275, 116), (420, 129)])
car = np.load("work/run2_car_raw.npy"); car = car[car[:, 1] >= 1.57]
t = car[:, 1]; car_ref = car[:, 2:4] - off                  # panorama px -> ref px (bottom-of-blob points)
def x_at(G, seg, y):                                         # where does a line cross lateral position y?
    p = G.to_lane(seg); a = (y - p[0, 1]) / (p[1, 1] - p[0, 1]); return p[0, 0] + a * (p[1, 0] - p[0, 0])
res = {}
for roll in [3.3, 5.1, 7.6]:
    G = Ground(f_ref, V_lane, roll)
    P = G.to_lane(car_ref); y_car = np.median(P[:, 1])
    xs, xf = x_at(G, start_line, y_car), x_at(G, finish_line, y_car)
    s = 75.0 / (xf - xs); x = s * (P[:, 0] - xs)
    X, S = rts_xva(t, x, np.full(len(t), 0.1), 20)
    res[roll] = X; print(f"roll {roll:4.1f}: camera height {s:5.2f} m, x at t=2,4,6 s: {np.interp([2, 4, 6], t, X[:, 0]).round(2)}, "
                          f"v at 2,4,6 s: {np.interp([2, 4, 6], t, X[:, 1]).round(2)}")
for r in [3.3, 7.6]:
    dv = res[r][:, 1] - res[5.1][:, 1]; da = (res[r][:, 2] - res[5.1][:, 2]) / 9.81
    print(f"roll {r} vs 5.1: max |dx| {np.abs(res[r][:,0]-res[5.1][:,0]).max():.2f} m, max |dv| {np.abs(dv).max():.2f} m/s, max |da| {np.abs(da).max():.3f} g")
