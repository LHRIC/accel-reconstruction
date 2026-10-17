# Registration check: detected cones in each frame vs the static cone map projected into that frame.
# The local offset near the car = registration error the car track inherits (same for both tyres).
import cv2, numpy as np, run2_setup as S
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
cm = np.load("work/run2_cone_map.npy")[:, :2] - S.OFF
d = np.load("work/run2_kin.npz"); tpx = d["tyre_px"]; k = float(d["k"]); T0 = float(d["t_launch"])
kf = np.load("work/run2_kf.npz")["kf"]
G = S.ground(h=k)
def detect(f):
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    m = (((hsv[..., 0] < 12) | (hsv[..., 0] > 168)) & (hsv[..., 1] > 110) & (hsv[..., 2] > 80)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    return np.array([(x + w / 2, y + h) for x, y, w, h, a in st[1:] if 4 <= a <= 4000 and h >= 3 and w <= 2.5 * h])
rows = []
for i in range(len(tpx)):
    f = cv2.imread(f"work/run2_jpg/{i:05d}.jpg"); det = detect(f)
    car = tpx[i, 1] if np.all(np.isfinite(tpx[i, 1])) else np.nanmean(tpx[i], axis=0)
    if len(det) == 0 or not np.all(np.isfinite(car)): rows.append((i, np.nan, np.nan, 0)); continue
    proj = S.from_ref(i, cm)
    res = []
    for p in proj:
        if not (0 <= p[0] < 640 and 0 <= p[1] < 360): continue
        dd = np.hypot(*(det - p).T); j = np.argmin(dd)
        if dd[j] < 5 and np.hypot(*(p - car)) < 120: res.append(det[j] - p)
    if len(res) >= 2:
        r = np.median(res, axis=0)
        # what this pixel offset means along the lane at the car
        a = G.to_lane(S.to_ref(i, [car, car + r]), z=S.TYRE_CENTRE_Z)[:, 0]
        rows.append((i, r[0], r[1], a[1] - a[0]))
    else: rows.append((i, np.nan, np.nan, np.nan))
rows = np.array(rows, float); np.save("work/run2_reg_residuals.npy", rows)
t = S.T_FRAME[:len(rows)] - T0
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].plot(t, rows[:, 1], ".", ms=3, label="du"); ax[0].plot(t, rows[:, 2], ".", ms=3, label="dv"); ax[0].set_ylabel("cone residual near car [px]")
ax[1].plot(t, rows[:, 3], "k.", ms=3); ax[1].set_ylabel("= along-lane error at car [m]")
for a_ in ax:
    for kk in kf: a_.axvline(S.T_FRAME[kk] - T0, color="r", lw=0.5, alpha=0.5)
    a_.grid(alpha=0.3)
ax[0].legend(); ax[1].set_xlabel("time since launch [s]   (red = keyframe switches)")
plt.tight_layout(); plt.savefig("review/reg_residuals.png", dpi=80)
print("median |along-lane error|: %.3f m, 90%%: %.3f m" % (np.nanmedian(np.abs(rows[:, 3])), np.nanpercentile(np.abs(rows[:, 3]), 90)))
