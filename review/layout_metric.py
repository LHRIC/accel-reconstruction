# Metric layout of run2 (roll 5.1 deg, scale from white lines = 75 m for now) - where is everything?
import numpy as np, sys
sys.path.insert(0, ".")
from calib3d import Ground
H0 = np.load("work/run2_kf.npz")["H_0i"]; Tm = np.load("work/run2_mosaicT.npy"); T = Tm[:9].reshape(3, 3); ref = int(Tm[9])
off = np.array([T[0, 2], T[1, 2]]); f_ref = float(np.load("work/run2_fref.npy")[0])
V_lane = np.load("work/run2_vps.npz")["V_lane"] - off
def to_ref(i, uv):
    M = np.linalg.inv(H0[ref]) @ H0[i]; uv = np.atleast_2d(uv); q = M @ np.c_[uv, np.ones(len(uv))].T
    return (q[:2] / q[2]).T
G = Ground(f_ref, V_lane, 5.1)
start_line = G.to_lane(to_ref(0, [(360, 106), (557, 132)])); finish_line = G.to_lane(to_ref(300, [(275, 116), (420, 129)]))
def x_at(p, y): a = (y - p[0, 1]) / (p[1, 1] - p[0, 1]); return p[0, 0] + a * (p[1, 0] - p[0, 0])
y_car = 0.0
car = np.load("work/run2_car_raw.npy"); car = car[car[:, 1] >= 1.57]; y_car = np.median(G.to_lane(car[:, 2:4] - off)[:, 1])
s = 75 / (x_at(finish_line, y_car) - x_at(start_line, y_car)); G.h = s
G.origin = np.zeros(3); x0 = x_at(G.to_lane(to_ref(0, [(360, 106), (557, 132)])), y_car * s)
print(f"camera height {s:.2f} m; car path lateral y = {y_car*s:.2f} m (lane coords, before origin shift)")
def show(name, lane_xy): print(f"  {name:34s} x = {lane_xy[0]-x0:7.2f} m   y = {lane_xy[1]:6.2f} m")
show("start white line @car path", (x0, y_car * s))
fl = G.to_lane(to_ref(300, [(275, 116), (420, 129)])); show("finish white line @car path", (x_at(fl, y_car * s), y_car * s))
show("start light base (f0)", G.to_lane(to_ref(0, [(390, 116)]))[0])
show("finish sign base (f300)", G.to_lane(to_ref(300, [(262, 114)]))[0])
show("staged car rear tyre bottom (f0)", G.to_lane(to_ref(0, [(296, 170)]))[0])
show("staged car front tyre bottom (f0)", G.to_lane(to_ref(0, [(347, 166)]))[0])
L = np.load("work/run2_lane.npz")
for nm, E in [("A", L["E1"]), ("B", L["E2"])]:
    P = G.to_lane(E - off); P = P[np.argsort(P[:, 0])]
    print(f"  cones {nm}: x =", (P[:, 0] - x0).round(1), " y =", P[:, 1].round(2).tolist()[:3], "...")
    print(f"     spacing: {np.diff(P[:, 0]).round(2)}")
