# Find straight line segments in the run2 panorama and sort them into
#  (1) lines along the lane  -> vanishing point V_lane
#  (2) near-vertical lines   -> vanishing point V_vert
import cv2, numpy as np
mos = cv2.imread("review/run2_mosaic.jpg"); g = cv2.cvtColor(mos, cv2.COLOR_BGR2GRAY)
Tm = np.load("work/run2_mosaicT.npy"); T = Tm[:9].reshape(3, 3); off = np.array([T[0, 2], T[1, 2]])
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
seg = lsd.detect(g)[0].reshape(-1, 4)                             # x1 y1 x2 y2 (panorama px)
L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]); seg = seg[L > 25]; L = L[L > 25]
print(len(seg), "segments > 25 px")
def hom_line(s):
    return np.cross([s[0], s[1], 1.0], [s[2], s[3], 1.0])
def vp_ransac(S, w, it=4000, tol_deg=0.6, seed=0):
    rng = np.random.default_rng(seed); best = None
    lines = np.array([hom_line(s) for s in S]); mids = (S[:, :2] + S[:, 2:]) / 2; dirs = S[:, 2:] - S[:, :2]
    def angerr(V):
        to_v = V[:2] / V[2] - mids if abs(V[2]) > 1e-9 else np.tile(V[:2], (len(S), 1))
        c = np.abs(np.sum(to_v * dirs, 1)) / (np.linalg.norm(to_v, axis=1) * np.linalg.norm(dirs, axis=1) + 1e-9)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))
    for _ in range(it):
        i, j = rng.choice(len(S), 2, replace=False); V = np.cross(lines[i], lines[j])
        if np.linalg.norm(V) < 1e-12: continue
        inl = angerr(V) < tol_deg
        if best is None or (w * inl).sum() > (w * best[1]).sum(): best = (V, inl)
    # least squares refine: V minimizing sum (line . V)^2 over inliers (lines normalized)
    Li = lines[best[1]]; Li = Li / np.linalg.norm(Li[:, :2], axis=1)[:, None]
    _, _, vt = np.linalg.svd(Li * np.sqrt(w[best[1]])[:, None]); V = vt[-1]
    return V, best[1], angerr(V)
ang = np.degrees(np.arctan2(seg[:, 3] - seg[:, 1], seg[:, 2] - seg[:, 0])) % 180
# lane family: lines that point roughly at the cone-line vanishing point
Vc = np.load("work/run2_lane.npz")["V"]
mids = (seg[:, :2] + seg[:, 2:]) / 2; dirs = seg[:, 2:] - seg[:, :2]; tov = Vc - mids
c = np.abs(np.sum(tov * dirs, 1)) / (np.linalg.norm(tov, axis=1) * np.linalg.norm(dirs, axis=1))
cand = np.degrees(np.arccos(np.clip(c, -1, 1))) < 3
V1, in1, e1 = vp_ransac(seg[cand], L[cand], tol_deg=0.5)
V1 = V1[:2] / V1[2]
print("V_lane (panorama px):", V1.round(1), " from", in1.sum(), "segments;  cone-line VP was", Vc.round(1))
vert = (np.abs(ang - 90) < 25) & (L > 30)
V3, in3, e3 = vp_ransac(seg[vert], L[vert], tol_deg=1.0)
V3 = V3[:2] / V3[2] if abs(V3[2]) > 1e-12 else V3[:2] * 1e9
print("V_vert (panorama px):", V3.round(1), " from", in3.sum(), "of", vert.sum(), "near-vertical segments")
np.savez("work/run2_vps.npz", V_lane=V1, V_vert=V3, off=off, seg_lane=seg[cand][in1], seg_vert=seg[vert][in3])
c0 = np.array([320.0, 180.0]) + off                            # ref-frame principal point in panorama px
f2 = -np.dot(V1 - c0, V3 - c0); print("f from orthogonality (lane vs vertical):", np.sqrt(f2).round(1) if f2 > 0 else f2)
vis = mos.copy()
for s in seg[cand][in1]: cv2.line(vis, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (0, 255, 255), 2)
for s in seg[vert][in3]: cv2.line(vis, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (255, 0, 255), 2)
cv2.imwrite("review/run2_vp_lines.jpg", vis[0:1300, 1500:4381])

# (3) transverse family: painted lines across the lane / pit-stall lines -> V_cross (should be on the horizon)
not_lane = ~cand
trans = not_lane & (ang > 0.5) & (ang < 20) & (L > 40)
V2, in2, e2 = vp_ransac(seg[trans], L[trans], tol_deg=0.6, seed=1)
V2 = V2[:2] / V2[2]
print("V_cross (panorama px):", V2.round(1), " from", in2.sum(), "of", trans.sum(), "candidate segments")
f2b = -np.dot(V1 - c0, V2 - c0); print("f from orthogonality (lane vs cross):", np.sqrt(f2b).round(1) if f2b > 0 else f2b)
np.savez("work/run2_vps.npz", V_lane=V1, V_cross=V2, off=off, seg_lane=seg[cand][in1], seg_cross=seg[trans][in2])
for s in seg[trans][in2]: cv2.line(vis, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (255, 128, 0), 3)
cv2.imwrite("review/run2_vp_lines.jpg", vis[0:1300, 1500:4381])
