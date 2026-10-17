# fine scan of f_ref (run2, ref frame 360) using every 4th frame of the run
import numpy as np, sys
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation as Rot
H0 = np.load("work/run2_kf.npz")["H_0i"]; ref = 360
c = np.array([320.0, 180, 1])
gx, gy = np.meshgrid(np.linspace(10, 630, 12), np.linspace(10, 350, 8)); G = np.c_[gx.ravel(), gy.ravel(), np.ones(gx.size)].T
def K(f): return np.array([[f, 0, 320.0], [0, f, 180.0], [0, 0, 1]])
def fit(H, f_ref):
    q = H @ G; q = q[:2] / q[2]
    w = H[2] @ c; qc = (H @ c) / w; J = (H[:2, :2] - np.outer(qc[:2], H[2, :2])) / w; zi = 1 / np.sqrt(abs(np.linalg.det(J)))
    M = np.linalg.inv(K(f_ref)) @ H @ K(f_ref * zi); U, S_, Vt = np.linalg.svd(M); R0 = U @ Vt
    if np.linalg.det(R0) < 0: R0 = -R0
    def res(x):
        m = K(f_ref) @ Rot.from_rotvec(x[:3]).as_matrix() @ np.linalg.inv(K(f_ref * np.exp(x[3]))) @ G
        return (m[:2] / m[2] - q).ravel()
    s = least_squares(res, np.r_[Rot.from_matrix(R0).as_rotvec(), np.log(zi)], method="lm")
    return np.mean(s.fun ** 2), s.x
Hs = [np.linalg.inv(H0[ref]) @ H0[i] for i in range(0, 400, 4)]
cost = lambda f: np.mean([fit(H, f)[0] for H in Hs])
r = minimize_scalar(cost, bounds=(2000, 2900), method="bounded", options=dict(xatol=5))
print(f"best f_ref = {r.x:.0f} px  (rms misfit {np.sqrt(r.fun):.2f} px)")
for f in [r.x * 0.9, r.x, r.x * 1.1]: print(f"  f={f:.0f}: rms {np.sqrt(cost(f)):.2f}px")
zooms = [np.exp(fit(H, r.x)[1][3]) for H in Hs]
print("implied f per frame (frames 0..400 step 40):", [round(r.x * z) for z in zooms[::10]])
np.save("work/run2_fref.npy", np.array([r.x]))
