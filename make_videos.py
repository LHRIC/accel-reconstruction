"""Post-processed videos for run2 (cut to the accel run):
  out/run2_steps.mp4      2x2: raw | ground grid + cones | SAM2 car/tyre tracking | bird's-eye position
  out/run2_telemetry.mp4  camera view with tracking + live x/v/a/RPM plots with a moving cursor
Original audio is kept (cut to match)."""
import cv2, numpy as np, json, subprocess, imageio_ffmpeg
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import run2_setup as S
from tyre_track import load_masks

d = np.load("work/run2_kin.npz"); r = json.load(open("work/run2_results.json"))
masks = load_masks(); k = float(d["k"]); x_st = float(d["x_staged"]); T0 = float(d["t_launch"])
G = S.ground(h=k)
frames = d["frames"]; t_kin = d["t"]
i0 = max(0, int(frames[0])); i1 = min(int(np.searchsorted(S.T_FRAME, r["finish_cross"] + 0.8)), len(masks) - 1)
FF = imageio_ffmpeg.get_ffmpeg_exe(); FPS = 59
L = np.load("work/run2_lane.npz"); cones = np.r_[L["E1"], L["E2"]] - S.OFF          # cone map, reference-frame px
yc = float(d["yc"])
START = x_st + S.WHEELBASE + 0.3                                                   # lane x of the start line


def kin_at(tq):
    return [np.interp(tq, t_kin, d[c]) for c in ("x", "v", "a")]


def lane_to_frame(i, pts_xyz):
    return S.from_ref(i, G.to_pixel(np.asarray(pts_xyz, float)))


def draw_poly(img, pts, col, th=1, s=2):
    p = np.round(pts * s).astype(int)
    for a, b in zip(p[:-1], p[1:]):
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), col, th, cv2.LINE_AA)


def ground_overlay(img, i, s=2):
    """lane edges + a line across the lane every 5 m (start/finish in green), drawn into frame i"""
    y_lo, y_hi = yc - 2.55, yc + 2.55
    for xm in np.arange(-5, 81, 5):
        col = (0, 255, 0) if xm in (0, 75) else (255, 255, 0)
        pts = lane_to_frame(i, [[START + xm, y, 0] for y in np.linspace(y_lo, y_hi, 8)])
        draw_poly(img, pts, col, 2 if xm in (0, 75) else 1, s)
        if xm % 10 == 0 or xm == 75:
            p = pts[0] * s
            if 0 < p[0] < img.shape[1] and 0 < p[1] < img.shape[0]:
                cv2.putText(img, f"{xm}m", (int(p[0]) + 3, int(p[1]) + 12), 0, 0.45, col, 1, cv2.LINE_AA)
    for y in (y_lo, y_hi):
        draw_poly(img, lane_to_frame(i, [[xl, y, 0] for xl in np.linspace(START - 8, START + 80, 60)]), (255, 255, 0), 1, s)


def track_overlay(img, i, s=2, trail=True):
    m = masks[i, 0]
    if m.any():
        col = np.zeros_like(img)
        col[cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0])) > 0] = (255, 0, 255)
        img[:] = cv2.addWeighted(img, 1.0, col, 0.35, 0)
    for j, c in enumerate([(0, 200, 255), (0, 255, 0)]):
        p = d["tyre_px"][i, j] + d["cone_off_s"][i]            # back to where the tyre is in the image
        if np.all(np.isfinite(p)):
            cv2.circle(img, (int(p[0] * s), int(p[1] * s)), 5, c, 2, cv2.LINE_AA)
    if trail:   # past rear-axle positions re-projected into this frame (shows the registration holds)
        for jj in range(max(i0, i - 120), i, 3):
            q = d["tyre_px"][jj]
            if np.any(np.isfinite(q)):
                p = (S.from_ref(i, S.to_ref(jj, np.nanmean(q, axis=0)))[0] + d["cone_off_s"][i]) * s
                cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 255), -1)


def hud(img, tq):
    x, v, a = kin_at(tq)
    lines = [f"t = {tq - T0:5.2f} s", f"x = {x:5.1f} m", f"v = {v:5.1f} m/s", f"a = {a / 9.81:+.2f} g"]
    for n_, txt in enumerate(lines):
        cv2.putText(img, txt, (12, 30 + 26 * n_), 0, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, txt, (12, 30 + 26 * n_), 0, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def build_birdseye(res=0.04):
    """top-down mosaic of the lane at the final geometry (median over frames)"""
    xs = np.arange(START - 6, START + 82, res); ys = np.arange(yc - 4.0, yc + 4.0, res)
    X, Y = np.meshgrid(xs, ys); uv = G.to_pixel(np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)])
    acc = []; cap = cv2.VideoCapture("videos/run2.mov"); i = 0
    while i < 400:
        ok, f = cap.read()
        if not ok:
            break
        if i % 4 == 0:
            q = S.from_ref(i, uv)
            mx = q[:, 0].reshape(X.shape).astype(np.float32); my = q[:, 1].reshape(X.shape).astype(np.float32)
            w = cv2.remap(f, mx, my, cv2.INTER_LINEAR).astype(np.float32)
            w[~((mx > 1) & (mx < 638) & (my > 1) & (my < 358))] = np.nan
            acc.append(w)
        i += 1
    top = np.nan_to_num(np.nanmedian(np.array(acc), axis=0)).astype(np.uint8)[::-1]
    return top, xs, ys[::-1]


BE, be_x, be_y = build_birdseye()
cv2.imwrite("out/run2_birdseye.png", BE)
RES = 0.04


def birdseye_panel(tq, W=640, H=360, span=24.0):
    x, v, a = kin_at(tq); xl = x + x_st                           # rear axle lane x
    c0 = int((xl - span * 0.35 - be_x[0]) / RES); c1 = c0 + int(span / RES)
    c0c, c1c = max(c0, 0), min(c1, BE.shape[1])
    strip = np.zeros((BE.shape[0], int(span / RES), 3), np.uint8)
    if c1c > c0c:
        strip[:, c0c - c0:c1c - c0] = BE[:, c0c:c1c]
    panel = cv2.resize(strip, (W, int(BE.shape[0] * W / strip.shape[1])))
    sc = W / strip.shape[1] / RES                                  # panel px per metre

    def to_p(xm, ym):
        return int((xm - (be_x[0] + c0 * RES)) * sc), int((be_y[0] - ym) * sc)
    cv2.rectangle(panel, to_p(xl - 0.5, yc + 0.7), to_p(xl + S.WHEELBASE + 0.6, yc - 0.7), (255, 0, 255), 2)
    cv2.circle(panel, to_p(xl, yc), 4, (0, 255, 255), -1)
    for xm in range(0, 80, 5):
        p = to_p(START + xm, yc + 3.2)
        cv2.line(panel, to_p(START + xm, yc + 3.6), to_p(START + xm, yc + 3.0), (255, 255, 0), 1)
        if 0 <= p[0] < W:
            cv2.putText(panel, f"{xm}", (p[0] - 8, p[1] + 16), 0, 0.4, (255, 255, 0), 1)
    out = np.zeros((H, W, 3), np.uint8); hh = min(H, panel.shape[0]); y0 = (H - hh) // 2
    out[y0:y0 + hh] = panel[:hh]
    hud(out, tq)
    return out


def label(img, txt):
    cv2.rectangle(img, (0, img.shape[0] - 24), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, img.shape[0] - 7), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def writer(path, W, H):
    t_a = S.T_FRAME[i0]; dur = (i1 - i0 + 1) / FPS
    cmd = [FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-ss", f"{t_a:.3f}", "-t", f"{dur:.3f}", "-i", "videos/run2.mov", "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "medium", "-c:a", "aac", "-b:a", "96k", "-shortest", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# ---- video 1: processing steps (2x2) ----
p = writer("out/run2_steps.mp4", 1280, 720); cap = cv2.VideoCapture("videos/run2.mov"); i = 0
while i <= i1:
    ok, f = cap.read()
    if not ok:
        break
    if i >= i0:
        tq = S.T_FRAME[i]
        a = f.copy(); label(a, "1  raw phone video (640x360, ~59 fps)")
        b = cv2.resize(f, (1280, 720)); ground_overlay(b, i)
        for c in S.from_ref(i, cones):
            cv2.circle(b, (int(c[0] * 2), int(c[1] * 2)), 6, (0, 0, 255), 1, cv2.LINE_AA)
        b = cv2.resize(b, (640, 360)); label(b, "2  ground plane: lane + 5 m lines (start/finish green), cone map red")
        c_ = cv2.resize(f, (1280, 720)); track_overlay(c_, i); c_ = cv2.resize(c_, (640, 360))
        label(c_, "3  SAM2 car mask + rear tyres, yellow trail = past axle positions")
        d_ = birdseye_panel(tq); label(d_, "4  top-down: rear axle (dot) + car box on reconstructed ground")
        p.stdin.write(np.vstack([np.hstack([a, b]), np.hstack([c_, d_])]).tobytes())
    i += 1
p.stdin.close(); p.wait()

# ---- video 2: telemetry ----
g = 9.81; tl = t_kin - T0; ta = d["te_audio"] - T0
fig, ax = plt.subplots(4, 1, figsize=(6.4, 7.2), dpi=100, sharex=True)
ax[0].plot(tl, d["x"], "k"); ax[0].set_ylabel("x [m]")
ax[1].plot(tl, d["v"], "k"); ax[1].set_ylabel("v [m/s]"); ax[1].set_ylim(0, 30)
ax[2].plot(tl, d["a"] / g, "k"); ax[2].set_ylabel("a [g]"); ax[2].set_ylim(-0.6, 1.6)
ax[3].plot(ta, d["rpm_true"] / 1000, color="tab:orange", lw=1); ax[3].set_ylabel("rpm / 1000"); ax[3].set_ylim(7, 14)
for a_ in ax:
    a_.grid(alpha=0.3)
    for s0, sm, s1 in r["shifts_accel"]:
        a_.axvline(s0 - T0, color="tab:blue", lw=1); a_.axvline(s1 - T0, color="tab:blue", lw=1, ls="--")
    for s0, s1 in r["shifts_audio"]:
        a_.axvline(s0 - T0, color="tab:orange", lw=1); a_.axvline(s1 - T0, color="tab:orange", lw=1, ls="--")
ax[3].set_xlabel("time since launch [s]  (blue = shift from video accel, orange = from audio; solid start, dashed end)", fontsize=7)
ax[0].set_xlim(S.T_FRAME[i0] - T0, S.T_FRAME[i1] - T0)
plt.tight_layout(); fig.canvas.draw()
plot_bg = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(), cv2.COLOR_RGB2BGR)
x_lo, x_hi = ax[0].get_xlim(); bb0 = ax[0].get_position(); bb3 = ax[3].get_position()


def t_to_px(tq):
    return int((bb0.x0 + (tq - x_lo) / (x_hi - x_lo) * (bb0.x1 - bb0.x0)) * plot_bg.shape[1])


y_top = int((1 - bb0.y1) * plot_bg.shape[0]); y_bot = int((1 - bb3.y0) * plot_bg.shape[0])
p = writer("out/run2_telemetry.mp4", 1280, 720); cap = cv2.VideoCapture("videos/run2.mov"); i = 0
while i <= i1:
    ok, f = cap.read()
    if not ok:
        break
    if i >= i0:
        tq = S.T_FRAME[i]
        left = cv2.resize(f, (1280, 720)); ground_overlay(left, i); track_overlay(left, i, trail=False)
        left = cv2.resize(left, (640, 360)); hud(left, tq)
        be = birdseye_panel(tq)
        right = plot_bg.copy(); xpx = t_to_px(tq - T0); cv2.line(right, (xpx, y_top), (xpx, y_bot), (0, 0, 255), 2)
        right = cv2.resize(right, (640, 720))
        p.stdin.write(np.hstack([np.vstack([left, be]), right]).tobytes())
    i += 1
p.stdin.close(); p.wait()
print("videos written: frames", i0, "to", i1)
