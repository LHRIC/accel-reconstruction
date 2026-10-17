# Frame-to-frame homographies (camera only rotates + zooms -> one 3x3 H per frame pair)
# Chain them to get every frame's mapping back to frame 0 (H_0i: frame i px -> frame 0 px)
import cv2, numpy as np, sys
run = int(sys.argv[1])
cap = cv2.VideoCapture(f"videos/run{run}.mov")
sift = cv2.SIFT_create(nfeatures=3000)
bf = cv2.BFMatcher(cv2.NORM_L2)
H_chain = [np.eye(3)]
n_inl, scale_step = [0], [1.0]
ok, f = cap.read()
kp0, d0 = sift.detectAndCompute(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), None)
while True:
    ok, f = cap.read()
    if not ok:
        break
    kp1, d1 = sift.detectAndCompute(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), None)
    H = np.eye(3); ni = 0
    if d0 is not None and d1 is not None and len(kp0) > 10 and len(kp1) > 10:
        m = [a for a, b in bf.knnMatch(d1, d0, k=2) if a.distance < 0.75 * b.distance]
        if len(m) >= 12:
            p1 = np.float32([kp1[x.queryIdx].pt for x in m]); p0 = np.float32([kp0[x.trainIdx].pt for x in m])
            Hc, mask = cv2.findHomography(p1, p0, cv2.USAC_MAGSAC, 1.5)
            if Hc is not None:
                H, ni = Hc, int(mask.sum())
    H_chain.append(H_chain[-1] @ H)
    n_inl.append(ni)
    scale_step.append(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
    kp0, d0 = kp1, d1
H_chain = np.array(H_chain)
np.savez(f"work/run{run}_cam.npz", H_0i=H_chain, n_inl=np.array(n_inl), scale_step=np.array(scale_step))
# zoom of frame i relative to frame 0 at image center (local scale of H_0i inverse)
print(f"run{run}: {len(H_chain)} frames, min inliers {min(n_inl[1:])}, frames<30 inl: {sum(np.array(n_inl[1:])<30)}")
