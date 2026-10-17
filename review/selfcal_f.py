# Focal-length self-calibration for a camera that only rotates (+zooms):
# every frame's homography to the reference frame must look like  K_ref * R * K_i^-1.
# Try a range of f_ref; for each, fit (R, f_i) per frame; the true f_ref gives the smallest misfit.
import numpy as np, sys
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
run = int(sys.argv[1]); ref = int(sys.argv[2]); i0, i1 = int(sys.argv[3]), int(sys.argv[4])
H0 = np.load(f"work/run{run}_kf.npz")["H_0i"]
gx, gy = np.meshgrid(np.linspace(10, 630, 12), np.linspace(10, 350, 8)); G = np.c_[gx.ravel(), gy.ravel(), np.ones(gx.size)].T
def K(f): return np.array([[f, 0, 320.0], [0, f, 180.0], [0, 0, 1]])
frames = list(range(i0, i1, 4))
Hs = [np.linalg.inv(H0[ref]) @ H0[i] for i in frames]
def misfit(f_ref):
    tot = []
    for H in Hs:
        q = H @ G; q = q[:2] / q[2]
        def res(x):
            m = K(f_ref) @ Rot.from_rotvec(x[:3]).as_matrix() @ np.linalg.inv(K(f_ref * np.exp(x[3]))) @ G
            return (m[:2] / m[2] - q).ravel()
        # initial guess: zoom from local scale at centre, rotation from nearest-rotation of K_ref^-1 H K_i
        c = np.array([320.0, 180, 1]); w = H[2] @ c; qc = (H @ c) / w
        J = (H[:2, :2] - np.outer(qc[:2], H[2, :2])) / w; zi = 1 / np.sqrt(abs(np.linalg.det(J)))
        M = np.linalg.inv(K(f_ref)) @ H @ K(f_ref * zi); U, S_, Vt = np.linalg.svd(M); R0 = U @ Vt
        if np.linalg.det(R0) < 0: R0 = -R0
        s = least_squares(res, np.r_[Rot.from_matrix(R0).as_rotvec(), np.log(zi)], method="lm")
        tot.append(np.sqrt(np.mean(s.fun ** 2)))
    return np.array(tot)
for f in [350, 450, 550, 650, 800, 1000, 1300]:
    e = misfit(f); print(f"f_ref={f:5d}: rms misfit median {np.median(e):.3f}px  mean {e.mean():.3f}px  max {e.max():.2f}px")
