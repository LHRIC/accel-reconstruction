# Sample panorama brightness along lines through the lane vanishing point near the car path; find painted dashes
import cv2, numpy as np
mos = cv2.imread("review/run2_mosaic.jpg"); g = cv2.cvtColor(mos, cv2.COLOR_BGR2GRAY).astype(float)
V = np.load("work/run2_vps.npz")["V_lane"]
car = np.load("work/run2_car_raw.npy"); car = car[car[:, 1] >= 1.57]
P0 = car[0, 2:4]                                      # first clean car point (panorama px)
best = None
for dy in np.arange(-40, 41, 2):                      # try lines through V and a point offset from the car path
    A = P0 + np.array([0, dy]); n = 3000
    pts = A[None] + (V - A)[None] * np.linspace(0, 0.97, n)[:, None]
    vals = cv2.remap(g.astype(np.float32), pts[:, 0].astype(np.float32).reshape(1, -1), pts[:, 1].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR)[0]
    bright = vals > np.percentile(vals, 50) + 45
    score = bright.sum()
    if best is None or score > best[0]: best = (score, dy, pts, vals, bright)
score, dy, pts, vals, bright = best
print("best offset", dy, "bright samples", score)
# runs of bright samples = dashes
edges = np.flatnonzero(np.diff(bright.astype(int)))
runs = [(edges[k] + 1, edges[k + 1]) for k in range(0, len(edges) - 1) if bright[edges[k] + 1]]
runs = [(a, b) for a, b in runs if b - a >= 3]
dash = np.array([[pts[a][0], pts[a][1], pts[b][0], pts[b][1]] for a, b in runs])
np.save("work/run2_dashes.npy", dash); print(len(dash), "dashes found")
vis = mos.copy()
for d in dash: cv2.line(vis, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (255, 0, 255), 3)
cv2.imwrite("review/run2_dashes.jpg", vis[100:900, 2200:4381])
