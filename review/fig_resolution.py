# How many metres along the lane does one video pixel cover where the car is? (sets measurement noise)
import cv2, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = np.load("work/run2_lane.npz"); H0 = np.load("work/run2_kf.npz")["H_0i"]; Tm = np.load("work/run2_mosaicT.npy")
T = Tm[:9].reshape(3, 3); ref = int(Tm[9]); d = np.load("work/run2_car_raw.npy"); lane = np.load("work/run2_car_lane.npy")
def ordered(E, m, dd):
    u = (E - m) @ dd; return E[np.argsort(u * np.sign(dd[0]))]
A = ordered(L["E1"], L["m1"], L["d1"]); B = ordered(L["E2"], L["m2"], L["d2"])
Hw, _ = cv2.findHomography(np.vstack([A, B]).astype(np.float32), np.vstack([np.c_[np.arange(len(A)), np.zeros(len(A))], np.c_[np.arange(len(B)), np.ones(len(B))]]).astype(np.float32), 0)
res, xs, ts = [], [], []
for r, l in zip(d, lane):
    if l[1] < 1.57: continue
    i = int(r[0]); M = Hw @ T @ np.linalg.inv(H0[ref]) @ H0[i]          # frame px -> lane coords
    p = np.linalg.inv(T @ np.linalg.inv(H0[ref]) @ H0[i]) @ np.array([r[2], r[3], 1.0]); p = p[:2] / p[2]
    def along(q):
        z = M @ np.array([q[0], q[1], 1.0]); return z[0] / z[2] * 5.0
    g = np.array([along(p + [0.5, 0]) - along(p - [0.5, 0]), along(p + [0, 0.5]) - along(p - [0, 0.5])])
    res.append(np.linalg.norm(g)); xs.append(l[2] * 5.0); ts.append(l[1])
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(xs, res, ".", ms=3); ax.set_xlabel("car position along lane [m] (provisional scale)")
ax.set_ylabel("metres along lane per video pixel"); ax.grid(alpha=0.3)
ax.set_title("run2: position resolution at the car (native frame pixels, includes zoom)")
plt.tight_layout(); plt.savefig("review/fig_resolution_run2.png", dpi=90)
print("m/px at x~5,20,40,60,75:", [round(float(np.interp(v, xs, res)), 3) for v in [5, 20, 40, 60, 75]])
