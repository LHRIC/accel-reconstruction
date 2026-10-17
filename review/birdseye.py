# Top-down (bird's-eye) mosaic of the lane on the reconstructed ground plane.
# For every output pixel (x along lane, y across) find where it lands in each frame, sample, combine.
import cv2, numpy as np, sys
sys.path.insert(0, ".")
from calib3d import Ground
run = 2; roll = float(sys.argv[1]) if len(sys.argv) > 1 else 5.1
V_shift = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0          # move the vanishing point up/down (px) to test tilt
tag = sys.argv[3] if len(sys.argv) > 3 else "base"
H0 = np.load("work/run2_kf.npz")["H_0i"]; Tm = np.load("work/run2_mosaicT.npy"); T = Tm[:9].reshape(3, 3); ref = int(Tm[9])
off = np.array([T[0, 2], T[1, 2]]); f_ref = float(np.load("work/run2_fref.npy")[0])
V = np.load("work/run2_vps.npz")["V_lane"] - off + np.array([0, V_shift])
G = Ground(f_ref, V, roll, h=7.0)
res = 0.05                                                         # m per output pixel
xs = np.arange(-30, 110, res); ys = np.arange(15.0, 26.0, res)
X, Y = np.meshgrid(xs, ys)
uv_ref = G.to_pixel(np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)])
acc = []; cap = cv2.VideoCapture("videos/run2.mov"); i = 0
while i < 400:
    ok, f = cap.read()
    if not ok: break
    if i % 3 == 0:
        M = np.linalg.inv(np.linalg.inv(H0[ref]) @ H0[i])          # ref px -> frame i px
        q = M @ np.c_[uv_ref, np.ones(len(uv_ref))].T; q = q[:2] / q[2]
        mx = q[0].reshape(X.shape).astype(np.float32); my = q[1].reshape(X.shape).astype(np.float32)
        w = cv2.remap(f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)).astype(np.float32)
        valid = (mx > 1) & (mx < 638) & (my > 1) & (my < 358)
        w[~valid] = np.nan; acc.append(w)
    i += 1
acc = np.array(acc)
top = np.nanmedian(acc, axis=0); top = np.nan_to_num(top).astype(np.uint8)
top = top[::-1]                                                  # so +y (far side, wall) is up
cv2.imwrite(f"work/run2_birdseye_{tag}.png", top)
np.savez(f"work/run2_birdseye_{tag}_grid.npz", xs=xs, ys=ys[::-1], roll=roll, V=V, f=f_ref)
print("bird's-eye", top.shape, "saved")
