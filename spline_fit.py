"""Fit a continuous cubic spline to the noisy velocity (finite differences of position), then take
acceleration as the spline's derivative. One figure per knot spacing:
    python spline_fit.py 0.5 0.7        (knot spacings in seconds)
Weights: 1/sigma of each point, so the far-away (noisier) points count less."""
import numpy as np, json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline

KNOT_SPACINGS = [float(a) for a in sys.argv[1:]] or [0.5, 0.7]

d = np.load("work/run2_kin.npz"); r = json.load(open("work/run2_results.json"))
T0 = float(d["t_launch"]); g = 9.81
t = d["t"] - T0                                    # time since launch [s]
v_raw, a_raw = d["v_raw"], d["a_raw"]               # central differences of position (no filtering)
sig_v = d["sig"] / np.median(np.diff(d["t"]))       # position noise / dt ~ velocity noise (relative weights)

keep = t > -0.25                                    # fit on the whole track (past the finish) so the
t, v_raw, a_raw, w = t[keep], v_raw[keep], a_raw[keep], 1 / sig_v[keep]   # spline ends don't flap in the plot
t_show = r["finish_cross"] - T0 + 0.4               # ...and only show up to just after the finish
sh = t <= t_show
tf = np.linspace(t[0], t_show, 2000)


def fit(dk, w):
    inner = np.arange(t[0] + dk, t[-1] - dk / 2, dk)
    knots = np.r_[[t[0]] * 4, inner, [t[-1]] * 4]   # cubic: end knots repeated 4x
    return make_lsq_spline(t, v_raw, knots, k=3, w=w)


for dk in KNOT_SPACINGS:
    ww = w.copy()
    for _ in range(2):                              # robust: drop points > 3 sigma off the fit, refit
        res = (v_raw - fit(dk, ww)(t)) * w
        ww = np.where(np.abs(res) > 3 * 1.4826 * np.median(np.abs(res)), 0.0, w)
    sp = fit(dk, ww)

    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(t[sh], v_raw[sh], ".", ms=3, color="0.55", label="noisy velocity (finite differences)")
    ax[0].plot(tf, sp(tf), "k", lw=2, label=f"cubic spline, knots every {dk:.2f} s")
    ax[1].plot(t[sh], a_raw[sh] / g, ".", ms=3, color="0.7", label="noisy accel (finite differences)")
    ax[1].plot(tf, sp.derivative()(tf) / g, "k", lw=2, label="spline derivative")
    for s0, s1 in r["shifts_audio"]:                # where the engine audio says the shifts are (for reference)
        for a_ in ax:
            a_.axvspan(s0 - T0, s1 - T0, color="tab:orange", alpha=0.15, lw=0)
    ax[0].set_ylabel("vx [m/s]"); ax[0].set_ylim(-3, 35); ax[0].legend(fontsize=8, loc="upper left")
    ax[1].set_ylabel("ax [g]"); ax[1].set_ylim(-1.0, 2.0); ax[1].legend(fontsize=8, loc="upper right")
    ax[1].set_xlabel("time since launch [s]   (orange bands = shifts from engine audio; raw accel scatter clipped, it spans +-6 g)")
    ax[0].set_title(f"run2: spline fit to noisy velocity, knots every {dk:.2f} s; acceleration = spline derivative", fontsize=10)
    for a_ in ax: a_.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"out/run2_spline_{dk:.2f}s.png".replace(".", "p", 1), dpi=90); plt.close()

    a_s = sp.derivative()(tf) / g
    print(f"knots {dk:.2f}s: {np.sum(ww == 0)} of {len(t)} points ignored as outliers, "
          f"velocity residual rms {np.sqrt(np.mean((v_raw[sh] - sp(t[sh])) ** 2)):.2f} m/s, "
          f"accel {a_s.min():+.2f} to {a_s.max():+.2f} g")
