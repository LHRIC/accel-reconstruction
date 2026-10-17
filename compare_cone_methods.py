"""Compare ways of locating cones / the ground plane (run2), as a short video + a plot.
Video panels:
  A  cones detected in each frame on their own (colour blobs, base point)
  B  cone map built from ALL frames, re-projected into each frame + straight-line fit through each edge
  C  ground grid (5 m) with tilt from the painted dashes   (V_SHIFT = -20 px)
  D  ground grid (5 m) with tilt from 'cones evenly spaced' (V_SHIFT = -35 px)
Plot: dash period, cone spacing and audio RPM/speed drift along the lane for each tilt option."""
import cv2, numpy as np, subprocess, imageio_ffmpeg, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import run2_setup as S
K_SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0      # camera height [m] (display only)

L = np.load("work/run2_lane.npz"); A_ref = L["E1"] - S.OFF; B_ref = L["E2"] - S.OFF
dash = np.load("work/run2_dash_ref_px.npz")["start"][:-1]


def detect_cones(f):
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    m = (((hsv[..., 0] < 12) | (hsv[..., 0] > 168)) & (hsv[..., 1] > 110) & (hsv[..., 2] > 80)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    return [(x + w / 2, y + h) for x, y, w, h, a in st[1:] if 4 <= a <= 4000 and h >= 3 and w <= 2.5 * h]


def edge_line_ref(E_ref, G):
    """straight line through one edge's cones, fitted on the ground, returned as 2 reference-frame points"""
    P = G.to_lane(E_ref); c = np.polyfit(P[:, 0], P[:, 1], 1)
    xs = np.array([P[:, 0].min() - 15, P[:, 0].max() + 30])
    return G.to_pixel(np.c_[xs, np.polyval(c, xs), np.zeros(2)])


def grid(img, i, G, start_x, yc, hw, col):
    for xm in np.arange(0, 91, 5):
        pts = S.from_ref(i, G.to_pixel(np.array([[start_x + xm, y, 0] for y in (yc - hw, yc + hw)]))) * 2
        cv2.line(img, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)), col, 1, cv2.LINE_AA)
    for y in (yc - hw, yc + hw):
        pts = S.from_ref(i, G.to_pixel(np.array([[start_x - 5, y, 0], [start_x + 95, y, 0]]))) * 2
        cv2.line(img, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)), col, 1, cv2.LINE_AA)
    for p in S.from_ref(i, dash) * 2:
        cv2.circle(img, tuple(p.astype(int)), 4, (255, 0, 255), 1, cv2.LINE_AA)


def label(img, txt):
    cv2.rectangle(img, (0, img.shape[0] - 24), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, img.shape[0] - 7), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


G_dash = S.ground(h=K_SCALE, v_shift=-20); G_cone = S.ground(h=K_SCALE, v_shift=-35)
# put each grid's 0 m line at the same place (first cone of edge A) so only the far-field differs
x0_d = G_dash.to_lane(A_ref)[:, 0].min(); x0_c = G_cone.to_lane(A_ref)[:, 0].min()
def lane_centre_halfwidth(G):
    ya, yb = np.median(G.to_lane(A_ref)[:, 1]), np.median(G.to_lane(B_ref)[:, 1])
    return (ya + yb) / 2, abs(yb - ya) / 2
y_d, hw_d = lane_centre_halfwidth(G_dash); y_c, hw_c = lane_centre_halfwidth(G_cone)
lines = [edge_line_ref(A_ref, G_dash), edge_line_ref(B_ref, G_dash)]

FF = imageio_ffmpeg.get_ffmpeg_exe()
cmd = [FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1280x720", "-r", "59", "-i", "-",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "out/run2_cone_methods.mp4"]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE); cap = cv2.VideoCapture("videos/run2.mov"); i = 0
while i < 400:
    ok, f = cap.read()
    if not ok:
        break
    if i >= 60:
        big = cv2.resize(f, (1280, 720))
        a = big.copy()
        for c in detect_cones(f):
            cv2.circle(a, (int(c[0] * 2), int(c[1] * 2)), 6, (0, 255, 255), 2, cv2.LINE_AA)
        b = big.copy()
        for E in (A_ref, B_ref):
            for c in S.from_ref(i, E) * 2:
                cv2.circle(b, tuple(c.astype(int)), 6, (0, 0, 255), 2, cv2.LINE_AA)
        for ln in lines:
            q = S.from_ref(i, ln) * 2
            cv2.line(b, tuple(q[0].astype(int)), tuple(q[1].astype(int)), (0, 255, 0), 1, cv2.LINE_AA)
        c_ = big.copy(); grid(c_, i, G_dash, x0_d, y_d, hw_d, (255, 255, 0))
        d_ = big.copy(); grid(d_, i, G_cone, x0_c, y_c, hw_c, (0, 165, 255))
        panels = [cv2.resize(x, (640, 360)) for x in (a, b, c_, d_)]
        label(panels[0], "A  cones found in this frame only (colour blobs)")
        label(panels[1], "B  cone map from all frames + straight edge-line fits")
        label(panels[2], "C  5 m grid, tilt from painted dashes (pink = dash ends)")
        label(panels[3], "D  5 m grid, tilt from 'cones evenly spaced'")
        p.stdin.write(np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]).tobytes())
    i += 1
p.stdin.close(); p.wait()

# ---- plot: spacing along the lane for each tilt option ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for vs, col in [(-5, "tab:red"), (-20, "tab:blue"), (-35, "tab:orange")]:
    G = S.ground(h=K_SCALE, v_shift=vs); x0 = G.to_lane(A_ref)[:, 0].min()
    xd = np.sort(G.to_lane(dash)[:, 0]) - x0
    ax[0].plot((xd[1:] + xd[:-1]) / 2, np.diff(xd) / np.diff(xd)[:3].mean(), "o-", color=col, label=f"tilt {vs:+d} px")
    for E, mk in ((A_ref, "s"), (B_ref, "^")):
        xc = np.sort(G.to_lane(E)[:, 0]) - x0
        ax[1].plot((xc[1:] + xc[:-1]) / 2, np.diff(xc) / np.diff(xc)[:4].mean(), mk + "-", color=col, ms=4, lw=0.8)
ax[0].set_title("painted dash period (machine-painted, should be flat)"); ax[1].set_title("cone gaps, edges A (squares) / B (triangles)")
for a_ in ax:
    a_.axhline(1, color="k", lw=0.8); a_.set_xlabel("distance down the lane [m]"); a_.set_ylabel("gap / gap near start"); a_.grid(alpha=0.3)
ax[0].legend()
plt.tight_layout(); plt.savefig("out/run2_tilt_options.png", dpi=90)
print("done")
