# Calibrate run2's ground plane: roll from 'start & finish lines are perpendicular to the lane',
# scale from start line -> finish line = 75 m.  Cone spacing only reported, not used.
import numpy as np, sys
sys.path.insert(0, ".")
from calib3d import Ground
H0 = np.load("work/run2_kf.npz")["H_0i"]; Tm = np.load("work/run2_mosaicT.npy"); T = Tm[:9].reshape(3, 3); ref = int(Tm[9])
off = np.array([T[0, 2], T[1, 2]])
f_ref = float(np.load("work/run2_fref.npy")[0])
V_lane = np.load("work/run2_vps.npz")["V_lane"] - off                 # panorama -> ref-frame px
def to_ref(i, uv):                                                     # frame i px -> ref-frame px
    M = np.linalg.inv(H0[ref]) @ H0[i]; q = M @ np.c_[np.atleast_2d(uv), np.ones(len(np.atleast_2d(uv)))].T
    return (q[:2] / q[2]).T
start_line = to_ref(0, [(360, 106), (557, 132)])        # white line under the start light (frame 0)
finish_line = to_ref(300, [(275, 116), (420, 129)])     # white line at the finish sign (frame 300)
L = np.load("work/run2_lane.npz"); coneA = L["E1"] - off; coneB = L["E2"] - off
car_lat = None
def angle_to_lane(G, seg):
    p = G.to_lane(seg); d = p[1] - p[0]; return np.degrees(np.arctan2(d[0], d[1]))   # 0 = perpendicular
print(f"f_ref = {f_ref:.0f} px;  lane VP (ref px) = {V_lane.round(1)}")
print(" roll  | start-line angle  finish-line angle (deg from perpendicular) | cone spacing A, B (unscaled)")
for roll in np.arange(-30, 31, 5):
    G = Ground(f_ref, V_lane, roll)
    sA = np.polyfit(np.arange(len(coneA)), np.sort(G.to_lane(coneA)[:, 0]), 1)[0]
    sB = np.polyfit(np.arange(len(coneB)), np.sort(G.to_lane(coneB)[:, 0]), 1)[0]
    print(f" {roll:5.1f} | {angle_to_lane(G, start_line):8.2f}  {angle_to_lane(G, finish_line):8.2f}            | {sA:.4f} {sB:.4f}  ratio {sA/sB:.4f}")
