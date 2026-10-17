"""run2: rear-tyre positions -> x(t), v(t), a(t).  Window 1 = raw finite differences, window 2 = filtered.
Shift markers: from the video acceleration and from the engine audio (start + end of each shift)."""
import numpy as np, json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import run2_setup as S
from tyre_track import load_masks, tyre_pixels, solve_scale, lane_positions, axle_x
sys.path.insert(0, "review"); from smoother import rts_xva

C_SOUND = 343.0
masks = load_masks(); n = len(masks); t = S.T_FRAME[:n]
tpx, tarea = tyre_pixels(masks)

# cones as local anchors: remove the local registration error near the car (python kinematics.py noanchor = off)
USE_CONE_ANCHOR = "noanchor" not in sys.argv
TAG = "" if USE_CONE_ANCHOR else "_noanchor"
from cone_anchor import local_offsets, smooth_offsets
car_px = np.where(np.isfinite(tpx[:, 1]), tpx[:, 1], tpx[:, 0])
cone_off = local_offsets(car_px); cone_off_s = smooth_offsets(cone_off)
if USE_CONE_ANCHOR:
    tpx = tpx - cone_off_s[:, None, :]

# ---- launch: first frame where the rear axle has moved > 5 cm from where it sat in frames 0-12 ----
STAGED = 13
k0, origin0, _ = solve_scale(tpx, t, launch_frame=STAGED)
xy0, _ = lane_positions(tpx, k0)
x0 = axle_x(xy0)
still = np.nanmedian(x0[:STAGED])
launch = int(np.flatnonzero((x0 - still > 0.05) & (np.arange(n) >= STAGED))[0])
k, origin, G = solve_scale(tpx, t, launch_frame=STAGED)
xy, _ = lane_positions(tpx, k)
x_rear = axle_x(xy)
filled = ~np.isfinite(xy[:, 1, 0]) & np.isfinite(xy[:, 0, 0])
print("frames filled from far-side tyre:", np.flatnonzero(filled).tolist())
x_staged = np.nanmedian(x_rear[:STAGED])
X = x_rear - x_staged                                                # distance travelled [m]
t_launch = t[launch]

# ---- per-frame measurement noise: metres along the lane per pixel at the tyre, x 0.5 px ----
sig = np.full(n, np.nan)
for i in range(n):
    p = np.nanmean(tpx[i], axis=0)
    if not np.all(np.isfinite(p)): continue
    ax_ = G.to_lane(S.to_ref(i, [p + [0.5, 0], p - [0.5, 0], p + [0, 0.5], p - [0, 0.5]]), z=S.TYRE_CENTRE_Z)[:, 0]
    sig[i] = 0.5 * np.hypot(ax_[0] - ax_[1], ax_[2] - ax_[3]) + 0.02

ok = np.isfinite(X) & np.isfinite(sig)
end = int(np.flatnonzero(ok)[-1])
sel = ok & (np.arange(n) <= end)
tt, xx, ss = t[sel], X[sel], sig[sel]

# ---- raw derivatives (no filtering): central differences of position ----
v_raw = np.gradient(xx, tt); a_raw = np.gradient(v_raw, tt)

# ---- filtered: RTS smoother (constant-acceleration model, random jerk) ----
# pass 1, then scale the noise model to the real scatter and down-weight outliers, pass 2
Q_JERK = 10.0
Xs, Ss = rts_xva(tt, xx, ss, Q_JERK)
z = (xx - Xs[:, 0]) / ss
ss = ss * 1.4826 * np.median(np.abs(z))                   # robust: make residuals ~ unit sigma
z = (xx - Xs[:, 0]) / ss
ss = np.where(np.abs(z) > 3.5, ss * 5, ss)                 # outliers count 25x less
print(f"noise scale {1.4826*np.median(np.abs((xx - Xs[:,0])/sig[sel])):.2f}, outliers down-weighted: {(np.abs(z) > 3.5).sum()}")
Xs, Ss = rts_xva(tt, xx, ss, Q_JERK)
xf, vf, af = Xs[:, 0], Xs[:, 1], Xs[:, 2]

# ---- timing check: front wheels cross start line (X = 0.3) and finish line (X = 75.3) ----
t_start = np.interp(0.3, xf, tt); t_fin = np.interp(75.3, xf, tt)
print(f"camera height {k:.2f} m | launch frame {launch} t={t_launch:.3f}s | start-line crossing {t_start:.3f}s, "
      f"finish {t_fin:.3f}s -> elapsed {t_fin - t_start:.3f}s (official {S.OFFICIAL_TIME})")

# ---- shift detection from video acceleration: dips in filtered a(t) ----
def accel_shifts(t_, a_, min_depth=1.0):
    out = []
    lo = [i for i in range(1, len(a_) - 1) if a_[i] < a_[i - 1] and a_[i] <= a_[i + 1]]
    for i in lo:
        j0 = i
        while j0 > 0 and a_[j0 - 1] >= a_[j0]: j0 -= 1          # preceding local max
        j1 = i
        while j1 < len(a_) - 1 and a_[j1 + 1] >= a_[j1]: j1 += 1  # following local max
        depth = min(a_[j0], a_[j1]) - a_[i]
        if depth < min_depth or t_[i] - t_launch < 0.6: continue
        half_in = (a_[j0] + a_[i]) / 2; half_out = (a_[j1] + a_[i]) / 2
        s0 = t_[j0 + np.flatnonzero(a_[j0:i + 1] <= half_in)[0]]
        s1 = t_[i + np.flatnonzero(a_[i:j1 + 1] >= half_out)[0]]
        out.append((s0, t_[i], s1))
    return out
fin_i = np.searchsorted(tt, t_fin)
shifts_accel = accel_shifts(tt[:fin_i], af[:fin_i])

# ---- shift detection from audio: RPM peak (start) -> RPM trough (end), moved to emission time ----
A = np.load("work/run2_rpm.npz"); ta, ra = A["t"], A["rpm_obs"]
def audio_shifts(ta, ra, min_drop=900):
    out = []; i = 0; dt = ta[1] - ta[0]; w = int(0.25 / dt)
    while i < len(ra) - w:
        if ra[i] - ra[i:i + w].min() > min_drop and ra[i] >= ra[max(0, i - 5):i + 5].max():
            j = i + int(np.argmin(ra[i:i + w]))
            j_end = j
            while j_end < len(ra) - 5 and ra[j_end + 5] <= ra[j_end] + 20: j_end += 1   # until it climbs again
            out.append((ta[i], ta[j_end])); i = j_end + 1
        else: i += 1
    return out
yc = np.nanmedian(xy[:, :, 1])
def distance(tq):
    """camera -> car distance [m] at time tq (car on the ground, camera height k)"""
    xl = np.interp(tq, tt, xf) + x_staged
    return np.sqrt(xl ** 2 + yc ** 2 + k ** 2)
def emit_time(t_obs):
    te = t_obs
    for _ in range(5): te = t_obs - distance(te) / C_SOUND
    return te
shifts_audio_obs = audio_shifts(ta, ra)
shifts_audio = [(emit_time(a), emit_time(b)) for a, b in shifts_audio_obs]

# true RPM (Doppler removed) at emission time, for the export
te_a = np.array([emit_time(x) for x in ta])
vr = np.interp(te_a, tt, vf) * (np.interp(te_a, tt, xf) + x_staged) / distance(te_a)
rpm_true = ra * (C_SOUND + vr) / C_SOUND

res = dict(camera_height=k, launch_t=float(t_launch), start_cross=float(t_start), finish_cross=float(t_fin),
           elapsed=float(t_fin - t_start), shifts_accel=shifts_accel, shifts_audio=shifts_audio, shifts_audio_obs=shifts_audio_obs,
           q_jerk=Q_JERK)
json.dump(res, open(f"work/run2_results{TAG}.json", "w"), indent=1, default=float)
np.savez(f"work/run2_kin{TAG}.npz", cone_off=cone_off, cone_off_s=cone_off_s, t=tt, x_raw=xx, sig=ss, v_raw=v_raw, a_raw=a_raw, x=xf, v=vf, a=af, Ss=Ss,
         t_launch=t_launch, x_staged=x_staged, k=k, tyre_px=tpx, frames=np.flatnonzero(sel), ta=ta, rpm_obs=ra,
         te_audio=te_a, rpm_true=rpm_true, yc=yc)
print("accel shifts (start, min, end):", [tuple(round(v, 3) for v in s) for s in shifts_accel])
print("audio shifts emission (start, end):", [tuple(round(v, 3) for v in s) for s in shifts_audio])
print("audio shifts as heard (start, end):", [tuple(round(v, 3) for v in s) for s in shifts_audio_obs])
