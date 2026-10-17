# Find the lane-centre dashes in the bird's-eye mosaic, convert their ends back to reference-frame pixels
import cv2, numpy as np, sys
sys.path.insert(0, ".")
from calib3d import Ground
top = cv2.imread("work/run2_birdseye_base.png"); g = np.load("work/run2_birdseye_base_grid.npz")
xs, ys = g["xs"], g["ys"]; G = Ground(float(g["f"]), g["V"], float(g["roll"]), h=7.0)
gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY).astype(float)
valid = top.sum(2) > 0
# find rows that contain the dashes: rows with strongest bright-dash energy
rows = np.flatnonzero((ys > 22.1) & (ys < 22.8)); r0 = int(rows.mean()); print("dash row y =", ys[r0].round(2), "m")
band = gray[rows].max(0); vb = valid[rows].all(0)
bg = np.median(band[vb]); bright = (band > bg + 35) & vb
e = np.flatnonzero(np.diff(bright.astype(int)))
runs = [(e[k] + 1, e[k + 1]) for k in range(len(e) - 1) if bright[e[k] + 1]]
runs = [(a, b) for a, b in runs if (b - a) * 0.05 > 0.8]           # dashes longer than 0.8 m
d0 = np.array([xs[a] for a, b in runs]); d1 = np.array([xs[b] for a, b in runs])
print("dash starts [m]:", d0.round(2)); print("dash lengths [m]:", (d1 - d0).round(2)); print("gaps start-to-start:", np.diff(d0).round(2))
uv0 = G.to_pixel(np.c_[d0, np.full(len(d0), ys[r0]), np.zeros(len(d0))]); uv1 = G.to_pixel(np.c_[d1, np.full(len(d1), ys[r0]), np.zeros(len(d1))])
np.savez("work/run2_dash_ref_px.npz", start=uv0, end=uv1)
