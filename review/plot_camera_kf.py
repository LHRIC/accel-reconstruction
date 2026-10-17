# Corrected camera zoom/pan vs time (keyframe registration). Zoom > 1 = zoomed in relative to frame 0.
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
c = np.array([320.0, 180.0, 1.0])
def zoom_pan(H):
    w = H[2] @ c; q = (H @ c) / w; J = (H[:2, :2] - np.outer(q[:2], H[2, :2])) / w
    return 1 / np.sqrt(abs(np.linalg.det(J))), q[:2]
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=False)
for k, run in enumerate([1, 2, 3, 4]):
    H = np.load(f"work/run{run}_kf.npz")["H_0i"]; t = np.load(f"frames/run{run}_t.npy")[:len(H)]
    zp = [zoom_pan(h) for h in H]; z = np.array([a for a, b in zp])
    z[(z > 50) | (z < 0.02)] = np.nan                   # after the camera turns away completely the numbers are meaningless
    axs[k].plot(t, z, "k"); axs[k].set_ylabel(f"run{run}\nzoom vs frame 0"); axs[k].grid(alpha=0.3)
    axs[k].set_yscale("log"); axs[k].set_yticks([0.5, 1, 2, 4, 8]); axs[k].set_yticklabels(["0.5", "1", "2", "4", "8"])
axs[-1].set_xlabel("video time [s]")
axs[0].set_title("CORRECTED camera zoom (keyframe registration). Earlier plot used drifting frame-to-frame chaining.", fontsize=9)
plt.tight_layout(); plt.savefig("out/camera_zoom_corrected.png", dpi=80)
