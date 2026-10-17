# Split cone map into the two lane edges (RANSAC lines), get vanishing point,
# and test "evenly spaced cones" with a 1D projective fit along each edge.
import numpy as np
c = np.load("work/run2_cone_map.npy")[:, :2]
c = c[(c[:, 0] > 1700 + 590 * 1.34)]                     # lane region only (drop start-box clutter)
rng = np.random.default_rng(0)
def ransac_line(P, tol=4.0, it=3000):
    best = None
    for _ in range(it):
        a, b = P[rng.choice(len(P), 2, replace=False)]
        d = b - a; n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        inl = np.abs((P - a) @ n) < tol
        if best is None or inl.sum() > best.sum(): best = inl
    Q = P[best]; m = Q.mean(0); u, s, vt = np.linalg.svd(Q - m); return best, m, vt[0]
in1, m1, d1 = ransac_line(c); rest = c[~in1]
in2, m2, d2 = ransac_line(rest)
E1, E2 = c[in1], rest[in2]
# vanishing point = intersection of the two edge lines
A = np.array([d1, -d2]).T; s = np.linalg.solve(A, m2 - m1); V = m1 + s[0] * d1
print("edge A cones", len(E1), " edge B cones", len(E2), " vanishing pt", V.round(1))
np.savez("work/run2_lane.npz", E1=E1, E2=E2, V=V, m1=m1, d1=d1, m2=m2, d2=d2)
# 1D projective fit: index k = (a*u + b) / (c*u + 1), u = coordinate along the edge line
for name, E, m, d in [("A", E1, m1, d1), ("B", E2, m2, d2)]:
    u = (E - m) @ d; order = np.argsort(u * np.sign(d[0])); u = u[order]
    k = np.arange(len(u)).astype(float)
    # linear least squares for a,b,cc in k*(cc*u+1) = a*u + b
    M = np.c_[u, np.ones_like(u), -k * u]; a, b, cc = np.linalg.lstsq(M, k, rcond=None)[0]
    kfit = (a * u + b) / (cc * u + 1)
    print(f"edge {name}: {len(u)} cones, evenly-spaced fit residual (in cone spacings): "
          f"rms {np.sqrt(np.mean((kfit - k) ** 2)):.3f}, max {np.abs(kfit - k).max():.3f}")
    print("   residuals:", np.round(kfit - k, 2))
