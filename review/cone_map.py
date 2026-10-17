# Build a cone map in panorama coords: detect cones in every frame at native res, map base points
# into the panorama, keep spots that are hit consistently (cones are static; vests/people move)
import cv2, numpy as np, sys
run, i0, i1 = 2, 0, 400
H0 = np.load(f"work/run{run}_kf.npz")["H_0i"]; Tm = np.load(f"work/run{run}_mosaicT.npy")
T = Tm[:9].reshape(3, 3); ref = int(Tm[9])
cap = cv2.VideoCapture(f"videos/run{run}.mov"); pts = []; i = 0
while i < i1:
    ok, f = cap.read()
    if not ok: break
    if i >= i0 and i % 2 == 0:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        m = (((hsv[..., 0] < 12) | (hsv[..., 0] > 168)) & (hsv[..., 1] > 110) & (hsv[..., 2] > 80)).astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        M = T @ np.linalg.inv(H0[ref]) @ H0[i]
        for k in range(1, n):
            x, y, w, h, a = st[k]
            if 4 <= a <= 4000 and h >= 3 and w <= 2.5 * h:
                q = M @ np.array([x + w / 2, y + h, 1.0]); pts.append((q[0] / q[2], q[1] / q[2], i))
    i += 1
pts = np.array(pts)
# cluster: 2D histogram in panorama px, keep dense cells, then merge neighbours
from scipy import ndimage
H2, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=[np.arange(0, 4400, 3), np.arange(0, 1820, 3)])
H2s = ndimage.uniform_filter(H2, 3) * 9
lab, n = ndimage.label(H2s > 12)
cones = []
for k in range(1, n + 1):
    sl = lab == k; w = H2s * sl
    cx = (w.sum(1) @ (xe[:-1] + 1.5)) / w.sum(); cy = (w.sum(0) @ (ye[:-1] + 1.5)) / w.sum()
    cones.append((cx, cy, H2[sl].sum()))
cones = np.array(cones); np.save("work/run2_cone_map.npy", cones)
print(len(pts), "detections ->", len(cones), "static cone spots")
mos = cv2.imread("review/run2_mosaic.jpg"); vis = mos.copy()
for cx, cy, c in cones:
    cv2.circle(vis, (int(cx), int(cy)), 7, (255, 255, 0), 2)
cv2.imwrite("review/run2_cone_map.jpg", vis[0:1300, 1700:4381])
