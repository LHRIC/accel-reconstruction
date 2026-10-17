# Zoom (relative to frame 0) and pan of the image center, from the chained homographies
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
c = np.array([320.0, 180.0, 1.0])
def local_scale(H, p):
    q = H @ p; w = q[2]; q = q / w
    J = (H[:2, :2] - np.outer(q[:2], H[2, :2])) / w
    return np.sqrt(abs(np.linalg.det(J))), q[:2]
fig, axs = plt.subplots(4, 2, figsize=(13, 11), sharex="row")
for k, run in enumerate([1, 2, 3, 4]):
    d = np.load(f"work/run{run}_cam.npz"); H = d["H_0i"]; t = np.load(f"frames/run{run}_t.npy")[:len(H)]
    s, ctr = zip(*[local_scale(h, c) for h in H])
    zoom = 1 / np.array(s); ctr = np.array(ctr)
    axs[k, 0].plot(t, zoom, "k"); axs[k, 0].set_ylabel(f"run{run}\nzoom vs frame0")
    axs[k, 1].plot(t, ctr[:, 0] - 320, label="x"); axs[k, 1].plot(t, ctr[:, 1] - 180, label="y")
    axs[k, 1].set_ylabel("view center shift\n[frame0 px]"); axs[k, 1].legend(loc="best", fontsize=8)
    axs[k, 1].set_ylim(-1500, 1500)
    print(run, "zoom range", zoom.min().round(2), zoom.max().round(2))
axs[-1, 0].set_xlabel("video time [s]"); axs[-1, 1].set_xlabel("video time [s]")
plt.tight_layout(); plt.savefig("review/camera_motion.png", dpi=80)
