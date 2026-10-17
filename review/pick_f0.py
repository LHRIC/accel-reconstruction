# Pick f0: compare rotation-model prediction vs direct SIFT homography between frames ~gap apart
import cv2, numpy as np, sys
sys.path.insert(0, "review"); from rotzoom import build, K, G
run = int(sys.argv[1]); gap = int(sys.argv[2]); f0s = [float(a) for a in sys.argv[3].split(",")]
idx = list(range(0, int(sys.argv[4]), 20))
cap = cv2.VideoCapture(f"videos/run{run}.mov"); frames = {}; i = 0
need = set(idx) | {j + gap for j in idx}
while True:
    ok, f = cap.read()
    if not ok: break
    if i in need: frames[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    i += 1
sift = cv2.SIFT_create(4000); bf = cv2.BFMatcher()
direct = {}
for a in idx:
    b = a + gap
    if b not in frames: continue
    ka, da = sift.detectAndCompute(frames[a], None); kb, db = sift.detectAndCompute(frames[b], None)
    m = [x for x, y in bf.knnMatch(db, da, k=2) if x.distance < 0.7 * y.distance]
    if len(m) < 30: continue
    pb = np.float32([kb[x.queryIdx].pt for x in m]); pa = np.float32([ka[x.trainIdx].pt for x in m])
    Hd, msk = cv2.findHomography(pb, pa, cv2.USAC_MAGSAC, 1.5)
    if Hd is not None and msk.sum() > 30: direct[(a, b)] = (pb[msk.ravel() > 0], pa[msk.ravel() > 0])
print("direct pairs:", len(direct))
for f0 in f0s:
    Rs, fs, _ = build(run, f0)
    errs = []
    for (a, b), (pb, pa) in direct.items():
        Hm = K(fs[a]) @ Rs[a].T @ Rs[b] @ np.linalg.inv(K(fs[b]))
        q = Hm @ np.c_[pb, np.ones(len(pb))].T; q = q[:2] / q[2]
        errs.append(np.median(np.linalg.norm(q.T - pa, axis=1)))
    print(f"f0={f0:6.0f}  median reproj err over {gap}-frame gaps: {np.median(errs):.2f}px  (90%: {np.percentile(errs,90):.2f})")
