# Stitch frames of one run into a panorama in the coordinates of a reference frame
import cv2, numpy as np, sys
run, ref, i0, i1, step = [int(a) for a in sys.argv[1:6]]
H0 = np.load(f"work/run{run}_kf.npz")["H_0i"]
H_ref_i = [np.linalg.inv(H0[ref]) @ H0[i] for i in range(len(H0))]  # frame i px -> ref px
# canvas bounds from frame corners
corners = np.array([[0, 0, 1], [639, 0, 1], [639, 359, 1], [0, 359, 1]], float).T
pts = []
for i in range(i0, i1, step):
    q = H_ref_i[i] @ corners; pts.append((q[:2] / q[2]).T)
pts = np.vstack(pts); lo = np.floor(pts.min(0)) - 5; hi = np.ceil(pts.max(0)) + 5
print("canvas", lo, hi)
T = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]])
W, Hh = int(hi[0] - lo[0]), int(hi[1] - lo[1])
acc = np.zeros((Hh, W, 3), np.float32); wsum = np.zeros((Hh, W), np.float32)
cap = cv2.VideoCapture(f"videos/run{run}.mov"); i = 0
while i < i1:
    ok, f = cap.read()
    if not ok: break
    if i >= i0 and (i - i0) % step == 0:
        M = T @ H_ref_i[i]
        w = cv2.warpPerspective(f.astype(np.float32), M, (W, Hh))
        m = cv2.warpPerspective(np.ones(f.shape[:2], np.float32), M, (W, Hh))
        acc += w * m[..., None]; wsum += m
    i += 1
mos = (acc / np.maximum(wsum, 1e-6)[..., None]).astype(np.uint8)
cv2.imwrite(f"review/run{run}_mosaic.jpg", mos)
np.save(f"work/run{run}_mosaicT.npy", np.array([*T.ravel(), ref]))
