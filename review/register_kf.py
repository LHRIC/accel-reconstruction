# Keyframe registration: match every frame to the current keyframe (not the previous frame),
# start a new keyframe when the match gets weak.  H_0i: frame i px -> frame 0 px.
import cv2, numpy as np, sys
run = int(sys.argv[1]); last = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
cap = cv2.VideoCapture(f"videos/run{run}.mov")
sift = cv2.SIFT_create(nfeatures=4000); bf = cv2.BFMatcher()
def feats(g): return sift.detectAndCompute(g, None)
def match(kp1, d1, kp0, d0):
    m = [a for a, b in bf.knnMatch(d1, d0, k=2) if a.distance < 0.75 * b.distance]
    if len(m) < 15: return None, 0
    p1 = np.float32([kp1[x.queryIdx].pt for x in m]); p0 = np.float32([kp0[x.trainIdx].pt for x in m])
    H, msk = cv2.findHomography(p1, p0, cv2.USAC_MAGSAC, 1.0)
    return (H, int(msk.sum())) if H is not None else (None, 0)
ok, f = cap.read(); g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
kf = feats(g); kf_H = np.eye(3); kf_inl0 = None
H_all = [np.eye(3)]; inl = [len(kf[0])]; kf_ids = [0]; prev = (kf, np.eye(3)); i = 0
while True:
    ok, f = cap.read(); i += 1
    if not ok or i > last: break
    cur = feats(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    H, n = match(cur[0], cur[1], kf[0], kf[1])
    if kf_inl0 is None: kf_inl0 = max(n, 1)
    if H is None or n < 120 or n < 0.35 * kf_inl0:
        # promote previous frame to keyframe, re-match current frame to it
        kf, kf_H = prev; kf_ids.append(i - 1)
        H, n = match(cur[0], cur[1], kf[0], kf[1]); kf_inl0 = max(n, 1)
        if H is None: H = np.eye(3); print("LOST at", i)
    Hi = kf_H @ H
    H_all.append(Hi); inl.append(n); prev = (cur, Hi)
np.savez(f"work/run{run}_kf.npz", H_0i=np.array(H_all), n_inl=np.array(inl), kf=np.array(kf_ids))
print(f"run{run}: {len(H_all)} frames, {len(kf_ids)} keyframes, min inliers {min(inl)}")
