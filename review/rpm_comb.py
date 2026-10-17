# Harmonic 'comb' score: for each candidate (observed) RPM, add up spectral energy at engine orders 2,4,6,8
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = np.load("work/run2_spec.npz"); f, t, SdB = d["f"], d["t"], d["SdB"]
# whiten each time slice: subtract a running median over frequency so loud broadband noise doesn't dominate
from scipy.ndimage import median_filter
W = SdB - median_filter(SdB, size=(301, 1))
rpm = np.arange(5000, 14501, 20.0)
orders = [2, 4, 6, 8]
score = np.zeros((len(rpm), len(t)))
for o in orders:
    fo = o * rpm / 60.0
    idx = np.clip(np.searchsorted(f, fo), 1, len(f) - 1)
    score += np.maximum(W[idx], W[idx - 1])
np.savez("work/run2_rpmcomb.npz", rpm=rpm, t=t, score=score)
plt.figure(figsize=(16, 6))
plt.pcolormesh(t, rpm, score, shading="auto", cmap="magma", vmin=np.percentile(score, 50), vmax=np.percentile(score, 99.8))
plt.ylabel("observed RPM (orders 2,4,6,8 combined)"); plt.xlabel("video time [s]"); plt.xticks(np.arange(0.5, 7.8, 0.25)); plt.grid(alpha=0.2)
plt.tight_layout(); plt.savefig("review/run2_rpm_comb.png", dpi=75)
