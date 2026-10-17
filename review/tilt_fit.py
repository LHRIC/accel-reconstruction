# Fix the ground-plane tilt: shift the lane vanishing point up/down (and try a few focal lengths) until the
# painted dashes are evenly spaced (paint = uniform). Then see whether the cones also come out even.
import numpy as np, sys
sys.path.insert(0, ".")
from calib3d import Ground
Tm = np.load("work/run2_mosaicT.npy"); T = Tm[:9].reshape(3, 3); off = np.array([T[0, 2], T[1, 2]])
f0 = float(np.load("work/run2_fref.npy")[0]); V0 = np.load("work/run2_vps.npz")["V_lane"] - off
dash = np.load("work/run2_dash_ref_px.npz"); D = dash["start"][:-1]          # drop the last (partial) dash
L = np.load("work/run2_lane.npz"); A = L["E1"] - off; B = L["E2"] - off
def curvature(x):           # fit x_k = a + b k + c k^2 ; return relative stretch over the run: 2c*N/b
    k = np.arange(len(x)); c2, b, a = np.polyfit(k, np.sort(x), 2); return 2 * c2 * (len(x) - 1) / b, b
print(" f_ref  Vshift | dash stretch  dash period | cone A stretch  cone B stretch | spacing A  B (m, h=7 scale)")
for f in [f0 * 0.9, f0, f0 * 1.1]:
    for vs in [-40, -30, -20, -10, 0, 10]:
        G = Ground(f, V0 + [0, vs], 5.1, h=7.0)
        sd, pd = curvature(G.to_lane(D)[:, 0]); sa, pa = curvature(G.to_lane(A)[:, 0]); sb, pb = curvature(G.to_lane(B)[:, 0])
        print(f" {f:5.0f}  {vs:5d}  |   {sd:+.3f}       {pd:5.2f}    |   {sa:+.3f}          {sb:+.3f}     | {pa:5.2f} {pb:5.2f}")
