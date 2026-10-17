# Preliminary run2 x/v/a: cone-spacing coordinates x 5.0 m (provisional scale), RTS smoothed
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, sys
sys.path.insert(0, "review"); from smoother import rts_xva
d = np.load("work/run2_car_lane.npy"); d = d[(d[:, 1] >= 1.57) & (d[:, 1] <= 6.75)]
DELTA = 5.0                                    # m per cone spacing (provisional, see notes)
t, x = d[:, 1], d[:, 2] * DELTA
# rough noise check: residual from a heavy smooth
Xs, _ = rts_xva(t, x, np.full(len(t), 0.3), q=50.0)
r = x - Xs[:, 0]; print("raw x residual rms vs smooth: %.3f m" % r.std())
sig = np.full(len(t), max(r.std(), 0.05))
fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
ax[0].plot(t, x, ".", ms=3, color="0.6", label="raw track (5 m/cone)")
v_raw = np.gradient(x, t)
ax[1].plot(t, v_raw, ".", ms=2, color="0.75", label="raw finite difference")
for q, col in [(20, "tab:blue"), (200, "tab:orange")]:
    Xs, S = rts_xva(t, x, sig, q)
    ax[0].plot(t, Xs[:, 0], color=col, lw=1.2, label=f"RTS smoother q={q}")
    ax[1].plot(t, Xs[:, 1], color=col, lw=1.5); ax[1].fill_between(t, Xs[:, 1] - 2 * S[:, 1], Xs[:, 1] + 2 * S[:, 1], color=col, alpha=0.15)
    ax[2].plot(t, Xs[:, 2] / 9.81, color=col, lw=1.5, label=f"q={q} (±2σ band)")
    ax[2].fill_between(t, (Xs[:, 2] - 2 * S[:, 2]) / 9.81, (Xs[:, 2] + 2 * S[:, 2]) / 9.81, color=col, alpha=0.15)
ax[0].set_ylabel("x [m]"); ax[1].set_ylabel("vx [m/s]"); ax[2].set_ylabel("ax [g]"); ax[2].set_xlabel("video time [s]")
ax[1].set_ylim(0, 35); ax[2].set_ylim(-0.5, 2.0)
ax[0].legend(fontsize=8); ax[1].legend(fontsize=8); ax[2].legend(fontsize=8)
for a in ax: a.grid(alpha=0.3)
ax[0].set_title("run2 — PRELIMINARY (single pass, provisional 5 m cone spacing, panorama resolution only)")
plt.tight_layout(); plt.savefig("review/run2_xva_prelim.png", dpi=90)
Xs, S = rts_xva(t, x, sig, 50)
for tt in [1.6, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5]:
    k = np.argmin(abs(t - tt)); print(f"t={t[k]:.2f}  x={Xs[k,0]:6.1f} m  v={Xs[k,1]:5.1f} m/s  a={Xs[k,2]/9.81:5.2f} g")
