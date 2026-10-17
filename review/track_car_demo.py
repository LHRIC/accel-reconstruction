# Quick car-tracking demo (run2): warp each frame into a lane-region canvas of the panorama,
# subtract a median background, take the biggest moving blob inside the lane.
import cv2, numpy as np
run, i0, i1 = 2, 0, 400
H0 = np.load(f"work/run{run}_kf.npz")["H_0i"]; Tm = np.load(f"work/run{run}_mosaicT.npy")
T = Tm[:9].reshape(3, 3); ref = int(Tm[9]); t_all = np.load(f"frames/run{run}_t.npy")
L = np.load("work/run2_lane.npz")
def ordered(E, m, d):
    u = (E - m) @ d; return E[np.argsort(u * np.sign(d[0]))]
A_ = ordered(L["E1"], L["m1"], L["d1"]); B_ = ordered(L["E2"], L["m2"], L["d2"])
Hw, _ = cv2.findHomography(np.vstack([A_, B_]).astype(np.float32), np.vstack([np.c_[np.arange(len(A_)), np.zeros(len(A_))], np.c_[np.arange(len(B_)), np.ones(len(B_))]]).astype(np.float32), 0)
X0, Y0, W, Hh = 2250, 150, 2131, 700                       # lane-region crop of the panorama
C = np.array([[1, 0, -X0], [0, 1, -Y0], [0, 0, 1.0]])
def M_of(i): return C @ T @ np.linalg.inv(H0[ref]) @ H0[i]
# lane polygon: between the two edge lines, widened a bit
def side(P, m, d): return (P[..., 0] - m[0]) * d[1] - (P[..., 1] - m[1]) * d[0]
yy, xx = np.mgrid[0:Hh, 0:W]; P = np.dstack([xx + X0, yy + Y0]).astype(float)
sA = side(P, L["m1"], L["d1"]); sB = side(P, L["m2"], L["d2"])
lane = (np.sign(sA) != np.sign(sB)) | (np.abs(sA) < 30) | (np.abs(sB) < 30)
lane &= (xx + X0) > 2250
cap = cv2.VideoCapture(f"videos/run{run}.mov"); warped, valid, idx = [], [], []; i = 0
while i < i1:
    ok, f = cap.read()
    if not ok: break
    if i >= i0:
        warped.append(cv2.warpPerspective(f, M_of(i), (W, Hh)))
        valid.append(cv2.warpPerspective(np.full(f.shape[:2], 255, np.uint8), M_of(i), (W, Hh)) > 250)
        idx.append(i)
    i += 1
stack = np.array(warped[::4]); vstack = np.array(valid[::4])
bg = np.zeros((Hh, W, 3), np.uint8)
for ch in range(3):                                        # median over frames where pixel is valid
    s = stack[..., ch].astype(np.float32); s[~vstack] = np.nan
    bg[..., ch] = np.nan_to_num(np.nanmedian(s, axis=0)).astype(np.uint8)
cv2.imwrite("work/run2_bg.jpg", bg)
rows = []
for k, i in enumerate(idx):
    d = np.abs(warped[k].astype(int) - bg.astype(int)).max(2)
    m = ((d > 40) & valid[k] & lane).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2: continue
    best = None
    for j in range(1, n):
        if st[j, 4] < 40: continue
        ys, xs = np.nonzero(lab == j)
        bot = ys >= np.percentile(ys, 85)
        q = cv2.perspectiveTransform(np.float32([[[xs[bot].mean() + X0, ys[bot].mean() + Y0]]]), Hw)[0, 0]
        if 0.25 < q[1] < 0.8 and (best is None or st[j, 4] > st[best, 4]): best = j
    if best is None: continue
    j = best; a = st[j, 4]
    ys, xs = np.nonzero(lab == j)
    bot = ys >= np.percentile(ys, 85)                      # lowest 15% of blob pixels ~ tyre contact
    rows.append((i, t_all[i], xs[bot].mean() + X0, ys[bot].mean() + Y0, cen[j][0] + X0, cen[j][1] + Y0, a))
rows = np.array(rows); np.save("work/run2_car_raw.npy", rows)
print(len(rows), "frames with a car blob; area range", rows[:, 6].min(), rows[:, 6].max())
vis = cv2.imread("review/run2_mosaic.jpg")
for r in rows[::3]:
    cv2.circle(vis, (int(r[2]), int(r[3])), 3, (0, 255, 0), -1)
cv2.imwrite("review/run2_track_on_mosaic.jpg", vis[150:850, 2250:4381])
