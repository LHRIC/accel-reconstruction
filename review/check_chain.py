# How much does chaining consecutive homographies drift? compare chain vs direct match over a gap
import cv2, numpy as np, sys
run = int(sys.argv[1]); gap = int(sys.argv[2]); last = int(sys.argv[3])
tag = sys.argv[4] if len(sys.argv) > 4 else "cam"
H0 = np.load(f"work/run{run}_{tag}.npz")["H_0i"]
cap = cv2.VideoCapture(f"videos/run{run}.mov"); frames = {}; i = 0
idx = list(range(0, last, 20)); need = set(idx) | {j + gap for j in idx}
while True:
    ok, f = cap.read()
    if not ok: break
    if i in need: frames[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    i += 1
sift = cv2.SIFT_create(4000); bf = cv2.BFMatcher()
for a in idx:
    b = a + gap
    if b not in frames: continue
    ka, da = sift.detectAndCompute(frames[a], None); kb, db = sift.detectAndCompute(frames[b], None)
    m = [x for x, y in bf.knnMatch(db, da, k=2) if x.distance < 0.7 * y.distance]
    if len(m) < 30: continue
    pb = np.float32([kb[x.queryIdx].pt for x in m]); pa = np.float32([ka[x.trainIdx].pt for x in m])
    Hd, msk = cv2.findHomography(pb, pa, cv2.USAC_MAGSAC, 1.5)
    pb, pa = pb[msk.ravel() > 0], pa[msk.ravel() > 0]
    Hc = np.linalg.inv(H0[a]) @ H0[b]
    q = Hc @ np.c_[pb, np.ones(len(pb))].T; q = q[:2] / q[2]
    qd = Hd @ np.c_[pb, np.ones(len(pb))].T; qd = qd[:2] / qd[2]
    print(f"{a:4d}->{b:4d}: inl {len(pb):4d}  chain err {np.median(np.linalg.norm(q.T - pa, axis=1)):6.2f}px  direct-H err {np.median(np.linalg.norm(qd.T - pa, axis=1)):5.2f}px")
