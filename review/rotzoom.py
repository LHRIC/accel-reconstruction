# Rotation + zoom camera model (camera stays in place, only turns and zooms):
#   pixel p = K_i R_i d   (d = ray direction in a fixed world frame), K_i = [[f_i,0,cx],[0,f_i,cy],[0,0,1]]
# Each consecutive pair's 8-DOF homography is re-fit with this 4-DOF model (3 rotation + 1 zoom),
# then chained.  f0 (focal of frame 0) is picked by checking against direct matches of far-apart frames.
import cv2, numpy as np, sys
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
cx, cy = 320.0, 180.0
gx, gy = np.meshgrid(np.linspace(20, 620, 9), np.linspace(20, 340, 6))
G = np.c_[gx.ravel(), gy.ravel(), np.ones(gx.size)].T          # grid pts in frame i

def K(f): return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])

def fit_pair(H, f_prev):
    """H maps frame i px -> frame i-1 px. Return (rotvec, f_i) so that K(f_prev) R K(f_i)^-1 ~ H"""
    q = H @ G; q = q[:2] / q[2]
    def res(x):
        R = Rot.from_rotvec(x[:3]).as_matrix(); fi = f_prev * np.exp(x[3])
        m = K(f_prev) @ R @ np.linalg.inv(K(fi)) @ G
        return (m[:2] / m[2] - q).ravel()
    s = least_squares(res, np.zeros(4))
    return s.x[:3], f_prev * np.exp(s.x[3]), np.sqrt(np.mean(s.fun ** 2))

def build(run, f0):
    H = np.load(f"work/run{run}_cam.npz")["H_0i"]
    Rs = [np.eye(3)]; fs = [f0]; rms = [0]
    for i in range(1, len(H)):
        Hpair = np.linalg.inv(H[i - 1]) @ H[i]          # frame i -> frame i-1
        rv, fi, r = fit_pair(Hpair, fs[-1])
        Rs.append(Rs[-1] @ Rot.from_rotvec(rv).as_matrix()); fs.append(fi); rms.append(r)
    return np.array(Rs), np.array(fs), np.array(rms)

if __name__ == "__main__":
    run = int(sys.argv[1]); f0 = float(sys.argv[2])
    Rs, fs, rms = build(run, f0)
    np.savez(f"work/run{run}_rotzoom.npz", R=Rs, f=fs, rms=rms, f0=f0)
    print(f"run{run} f0={f0}: pair-fit rms median {np.median(rms):.3f}px max {rms.max():.2f}px, f range {fs.min():.0f}-{fs.max():.0f}")
