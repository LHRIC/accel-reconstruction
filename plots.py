"""Two windows for run2: (1) raw x, v, a from finite differences; (2) filtered x, v, a with +-2 sigma.
Shift markers on both: blue = from video acceleration, orange = from engine audio; solid = start, dashed = end."""
import numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = np.load("work/run2_kin.npz"); r = json.load(open("work/run2_results.json"))
T0 = float(d["t_launch"]); t = d["t"] - T0                       # time since launch
g = 9.81
def marks(ax):
    for s0, smin, s1 in r["shifts_accel"]:
        ax.axvline(s0 - T0, color="tab:blue", lw=1.2); ax.axvline(s1 - T0, color="tab:blue", lw=1.2, ls="--")
    for s0, s1 in r["shifts_audio"]:
        ax.axvline(s0 - T0, color="tab:orange", lw=1.2); ax.axvline(s1 - T0, color="tab:orange", lw=1.2, ls="--")
    ax.axvspan(r["start_cross"] - T0, r["finish_cross"] - T0, color="0.9", zorder=-5)
def legend(ax):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="tab:blue", lw=1.2), Line2D([], [], color="tab:blue", lw=1.2, ls="--"),
         Line2D([], [], color="tab:orange", lw=1.2), Line2D([], [], color="tab:orange", lw=1.2, ls="--")]
    ax.legend(h, ["shift start (video accel)", "shift end (video accel)", "shift start (audio)", "shift end (audio)"],
              fontsize=8, loc="upper left", ncol=2)
tmax = r["finish_cross"] - T0 + 0.5
for kind in ["raw", "filtered"]:
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    if kind == "raw":
        ax[0].plot(t, d["x_raw"], ".", ms=2.5, color="k")
        ax[1].plot(t, d["v_raw"], ".", ms=2.5, color="k")
        ax[2].plot(t, d["a_raw"] / g, ".", ms=2.5, color="k")
        ax[1].set_ylim(-5, 40); ax[2].set_ylim(-6, 6)
        title = "run2 RAW: rear-tyre position per frame, v and a by finite differences (no filtering)"
    else:
        S = d["Ss"]
        ax[0].plot(t, d["x_raw"], ".", ms=2, color="0.6"); ax[0].plot(t, d["x"], color="k", lw=1.3)
        ax[1].fill_between(t, d["v"] - 2 * S[:, 1], d["v"] + 2 * S[:, 1], color="0.75"); ax[1].plot(t, d["v"], "k", lw=1.3)
        ax[2].fill_between(t, (d["a"] - 2 * S[:, 2]) / g, (d["a"] + 2 * S[:, 2]) / g, color="0.75"); ax[2].plot(t, d["a"] / g, "k", lw=1.3)
        A = np.load("work/run2_audio_va.npz")                       # independent: engine sound x gearing
        for j, key in enumerate(sorted(A.files)):
            tu, va, aa = A[key]
            ax[1].plot(tu - T0, va, color="tab:orange", lw=1.6, label="from engine audio (in gear)" if j == 0 else None)
            ax[2].plot(tu - T0, aa / g, color="tab:orange", lw=1.6)
        ax[1].legend(fontsize=8, loc="lower right")
        ax[1].set_ylim(0, 30); ax[2].set_ylim(-0.6, 1.6)
        title = (f"run2 FILTERED (video only): RTS smoother, jerk noise q={r['q_jerk']:g}, grey = +-2 sigma.  "
                 f"Orange = independent speed/accel from engine audio (2nd/3rd gear)")
    for a_ in ax: marks(a_); a_.grid(alpha=0.3)
    legend(ax[2])
    ax[0].set_ylabel("x travelled [m]"); ax[1].set_ylabel("vx [m/s]"); ax[2].set_ylabel("ax [g]")
    ax[2].set_xlabel(f"time since launch [s]   (shaded = timed section: start line {r['start_cross']-T0:.2f}s -> "
                     f"finish {r['finish_cross']-T0:.2f}s = {r['elapsed']:.3f}s, board says 4.837s)")
    ax[0].set_title(title, fontsize=10); ax[0].set_xlim(-0.3, tmax)
    plt.tight_layout(); plt.savefig(f"out/run2_{kind}.png", dpi=90); plt.close()
print("plots saved")
