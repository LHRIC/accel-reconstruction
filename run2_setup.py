"""Everything specific to run2 in one place (camera geometry + hand-picked reference points)."""
import numpy as np
from calib3d import Ground

RUN = 2
REF = 360                      # reference frame (everything is mapped into this frame's pixels)
F_REF = 2362.0                 # focal length of the reference frame [px], from rotation self-calibration
V_SHIFT = -20.0                # tilt fix: move lane vanishing point up 20 px so painted dashes are evenly spaced
ROLL = 5.1                     # plane roll about the lane axis [deg] (barely matters, see review)
TYRE_CENTRE_Z = 0.198          # rear tyre centre height = loaded radius [m] (8 in minus ~5 mm)
WHEELBASE = 1.535              # [m]
COURSE = 75.0                  # start line -> finish line [m]
OFFICIAL_TIME = 4.837          # timing board [s]

# hand-picked image points (frame, pixel) - see review/ crops
FINISH_LINE = (300, [(175.8, 152.3), (329.9, 169.1)])  # doubled-cone bases = finish (75 m from the staging line; see README)
CHECKERED_LINE = (300, [(275, 116), (420, 129)])   # painted line at the checkered sign (~100 m, not the timing line)
WHITE_START_LINE = (0, [(360, 106), (557, 132)])   # painted line under the green start light (~10 m, not the timing line)

H0 = np.load("work/run2_kf.npz")["H_0i"]                     # frame i px -> frame 0 px (keyframe registration)
_T = np.load("work/run2_mosaicT.npy")[:9].reshape(3, 3)
OFF = np.array([_T[0, 2], _T[1, 2]])                         # panorama px = reference px + OFF
V_LANE = np.load("work/run2_vps.npz")["V_lane"] - OFF + np.array([0.0, V_SHIFT])
T_FRAME = np.load("frames/run2_t.npy")                       # frame timestamps [s]

def to_ref(i, uv):
    """frame i pixel(s) -> reference-frame pixel(s)"""
    uv = np.atleast_2d(np.asarray(uv, float))
    M = np.linalg.inv(H0[REF]) @ H0[i]
    q = M @ np.c_[uv, np.ones(len(uv))].T
    return (q[:2] / q[2]).T

def from_ref(i, uv_ref):
    """reference-frame pixel(s) -> frame i pixel(s)"""
    uv_ref = np.atleast_2d(np.asarray(uv_ref, float))
    M = np.linalg.inv(np.linalg.inv(H0[REF]) @ H0[i])
    q = M @ np.c_[uv_ref, np.ones(len(uv_ref))].T
    return (q[:2] / q[2]).T

def ground(h=1.0, v_shift=V_SHIFT):
    return Ground(F_REF, np.load("work/run2_vps.npz")["V_lane"] - OFF + np.array([0.0, v_shift]), ROLL, h=h)

def line_x_at(G, frame_pts, y):
    """where a painted line (2 image points in one frame) crosses lateral position y on the ground"""
    i, pts = frame_pts
    p = G.to_lane(to_ref(i, pts))
    a = (y - p[0, 1]) / (p[1, 1] - p[0, 1])
    return p[0, 0] + a * (p[1, 0] - p[0, 0])
