"""3D ground-plane calibration for one run (camera only rotates/zooms, stays in one spot).

Everything is expressed in the reference frame's camera coordinates:
  ray for a reference-frame pixel (u, v):  r = [(u - 320)/f_ref, (v - 180)/f_ref, 1]
  ground plane:  n . X = -h   (n = unit 'up' normal, h = camera height above the ground)
  a ground point seen along ray r:  X = -h r / (n . r)
Lane frame: x along the lane (direction of travel), y across, origin on the start line.
"""
import numpy as np

class Ground:
    def __init__(self, f_ref, V_lane_ref, roll_deg, h=1.0):
        self.f = f_ref
        d1 = np.array([(V_lane_ref[0] - 320) / f_ref, (V_lane_ref[1] - 180) / f_ref, 1.0])
        self.ex = d1 / np.linalg.norm(d1)                    # along-lane direction (points away from camera)
        up0 = np.array([0.0, -1.0, 0.0])                     # image 'up' as first guess for the normal
        n0 = up0 - (up0 @ self.ex) * self.ex; n0 /= np.linalg.norm(n0)
        a = np.radians(roll_deg)                              # roll the plane about the lane direction
        self.n = n0 * np.cos(a) + np.cross(self.ex, n0) * np.sin(a)
        self.ey = np.cross(self.n, self.ex)                   # across-lane direction
        self.h = h; self.origin = np.zeros(3)

    def ray(self, uv_ref):
        uv_ref = np.atleast_2d(uv_ref)
        return np.c_[(uv_ref[:, 0] - 320) / self.f, (uv_ref[:, 1] - 180) / self.f, np.ones(len(uv_ref))]

    def to_lane(self, uv_ref, z=0.0):
        """reference-frame pixels -> lane coords (x, y) of the point at height z above the ground"""
        r = self.ray(uv_ref)
        X = -(self.h - z) * r / (r @ self.n)[:, None]
        X = X - self.origin
        return np.c_[X @ self.ex, X @ self.ey]

    def to_pixel(self, xyz):
        """lane coords (x, y, z) -> reference-frame pixel (for drawing overlays)"""
        xyz = np.atleast_2d(xyz)
        X = self.origin + xyz[:, :1] * self.ex + xyz[:, 1:2] * self.ey + (xyz[:, 2:3] - self.h) * self.n
        return np.c_[320 + self.f * X[:, 0] / X[:, 2], 180 + self.f * X[:, 1] / X[:, 2]]
