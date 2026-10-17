# Intensity profile along the painted dashed line (panorama), dash start/end points -> metric x via 3D model
import cv2, numpy as np, sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from calib3d import Ground
mos = cv2.imread("review/run2_mosaic.jpg"); g = cv2.cvtColor(mos, cv2.COLOR_BGR2GRAY).astype(np.float32)
d = np.load("work/run2_vps.npz"); V = d["V_lane"]; seg = d["seg_lane"]
mid = (seg[:, :2] + seg[:, 2:]) / 2; ang = np.degrees(np.arctan2(mid[:, 1] - V[1], mid[:, 0] - V[0]))
sel = seg[np.abs(ang - 166.684) < 0.06]
# refine the line through V: least squares angle using segment endpoints
pts = np.r_[sel[:, :2], sel[:, 2:]]; th = np.arctan2(pts[:, 1] - V[1], pts[:, 0] - V[0]); th0 = np.median(th)
u = np.array([np.cos(th0), np.sin(th0)]); nrm = np.array([-u[1], u[0]])
dist = np.arange(700, 3600, 0.5)                      # distance from V along the line (panorama px)
prof = []
for o in np.arange(-1.5, 1.6, 0.5):                   # average a thin band
    P = V[None] + dist[:, None] * u[None] + o * nrm[None]
    prof.append(cv2.remap(g, P[:, 0].reshape(1, -1).astype(np.float32), P[:, 1].reshape(1, -1).astype(np.float32), cv2.INTER_LINEAR)[0])
prof = np.max(prof, axis=0)
base = cv2.blur(prof.reshape(1, -1), (401, 1))[0]     # local background level
bright = prof > base + 25
e = np.flatnonzero(np.diff(bright.astype(int)))
runs = [(e[k] + 1, e[k + 1]) for k in range(len(e) - 1) if bright[e[k] + 1] and e[k + 1] - e[k] >= 4]
P_start = np.array([V + dist[b] * u for a, b in runs])   # end nearer the camera (larger distance from V)
P_end = np.array([V + dist[a] * u for a, b in runs])
print(len(runs), "dashes")
H0 = np.load("work/run2_kf.npz")["H_0i"]; Tm = np.load("work/run2_mosaicT.npy"); T = Tm[:9].reshape(3, 3)
off = np.array([T[0, 2], T[1, 2]]); f_ref = float(np.load("work/run2_fref.npy")[0])
G = Ground(f_ref, V - off, 5.1, h=6.98)
xs = G.to_lane(P_start - off)[:, 0]; xe = G.to_lane(P_end - off)[:, 0]
o = np.argsort(xs); xs, xe = xs[o], xe[o]
print("dash near-end x [m]:", xs.round(2))
print("dash length [m]:", (xe - xs).round(2))
print("period [m]:", np.diff(xs).round(2))
np.savez("work/run2_dash_pts.npz", P_start=P_start[o], P_end=P_end[o])
plt.figure(figsize=(12, 3)); plt.plot(dist, prof, lw=0.6); plt.plot(dist, base + 25, lw=0.6)
plt.xlabel("distance from vanishing point along dashed line [panorama px]"); plt.tight_layout(); plt.savefig("review/dash_profile.png", dpi=80)
