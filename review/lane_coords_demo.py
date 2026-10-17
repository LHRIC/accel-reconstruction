# Map tracked car ground point -> along-lane coordinate in "cone spacings" using a homography
# fit to the cones: edge A cone k -> (k, 0), edge B cone k -> (k, 1).  (Evenly spaced cones;
# any stagger between edges only shifts x by a constant, which doesn't change v or a.)
import cv2, numpy as np
L = np.load("work/run2_lane.npz"); rows = np.load("work/run2_car_raw.npy")
def ordered(E, m, d):
    u = (E - m) @ d; s = np.sign(d[0]); return E[np.argsort(u * s)]
A = ordered(L["E1"], L["m1"], L["d1"]); B = ordered(L["E2"], L["m2"], L["d2"])
src = np.vstack([A, B]).astype(np.float32)
dst = np.vstack([np.c_[np.arange(len(A)), np.zeros(len(A))], np.c_[np.arange(len(B)), np.ones(len(B))]]).astype(np.float32)
Hw, _ = cv2.findHomography(src, dst, 0)
res = cv2.perspectiveTransform(src[None], Hw)[0] - dst
print("cone homography fit rms (spacings):", np.sqrt((res ** 2).mean(0)).round(3))
P = cv2.perspectiveTransform(rows[None, :, 2:4].astype(np.float32), Hw)[0]
np.save("work/run2_car_lane.npy", np.c_[rows[:, 0], rows[:, 1], P])
for r, p in zip(rows[::20], P[::20]): print(f"frame {int(r[0]):4d} t={r[1]:6.3f}  along={p[0]:7.3f}  lateral={p[1]:6.3f}")
