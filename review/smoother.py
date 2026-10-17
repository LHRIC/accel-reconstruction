# Rauch-Tung-Striebel smoother for x(t) measurements -> x, v, a
# Model: constant acceleration + random jerk (q = jerk noise intensity [m^2/s^5]).
import numpy as np
def rts_xva(t, z, sig_z, q):
    n = len(t); X = np.zeros((n, 3)); P = np.zeros((n, 3, 3))
    Xp = np.zeros((n, 3)); Pp = np.zeros((n, 3, 3)); Fs = np.zeros((n, 3, 3))
    x = np.array([z[0], 0, 0.0]); Pk = np.diag([sig_z[0] ** 2, 100, 100.0])
    Hm = np.array([[1.0, 0, 0]])
    for k in range(n):
        if k > 0:
            dt = t[k] - t[k - 1]
            F = np.array([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1]])
            Q = q * np.array([[dt**5/20, dt**4/8, dt**3/6], [dt**4/8, dt**3/3, dt**2/2], [dt**3/6, dt**2/2, dt]])
            x = F @ x; Pk = F @ Pk @ F.T + Q; Fs[k] = F
        Xp[k], Pp[k] = x, Pk
        if np.isfinite(z[k]):
            S = Pk[0, 0] + sig_z[k] ** 2; K = Pk[:, 0] / S
            x = x + K * (z[k] - x[0]); Pk = Pk - np.outer(K, Hm @ Pk)
        X[k], P[k] = x, Pk
    Xs, Ps = X.copy(), P.copy()
    for k in range(n - 2, -1, -1):
        C = P[k] @ Fs[k + 1].T @ np.linalg.inv(Pp[k + 1])
        Xs[k] = X[k] + C @ (Xs[k + 1] - Xp[k + 1]); Ps[k] = P[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return Xs, np.sqrt(np.einsum("kii->ki", Ps))
