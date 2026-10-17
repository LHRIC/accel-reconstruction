"""Speed from the engine sound, gear by gear: v = RPM_true / (primary * final * gear) * 2*pi*r / 60.
RPM_true = heard RPM with the Doppler shift removed; each sample moved back to when the sound left the car.
Uses the video geometry only for the (small) Doppler and sound-delay corrections."""
import numpy as np
C = 343.0; R_TYRE = 0.198; PRIMARY = 2.111; FINAL = 2.84
GEARS = {1: 2.75, 2: 2.0, 3: 1.6667}
HEARD_WINDOWS = {2: (3.30, 4.28), 3: (4.72, 6.02)}   # clean in-gear stretches (heard time), away from shift transients


def audio_speed(kin):
    ta, ra = kin["ta"], kin["rpm_obs"]
    te_all = kin["te_audio"]                          # emission time of every audio sample (from kinematics.py)
    t, x, v = kin["t"], kin["x"] + kin["x_staged"], kin["v"]
    yc, k = float(kin["yc"]), float(kin["k"])
    out = {}
    for g, (a, b) in HEARD_WINDOWS.items():
        m = (ta >= a) & (ta <= b); te = te_all[m]
        dist = np.sqrt(np.interp(te, t, x) ** 2 + yc ** 2 + k ** 2)
        vr = np.interp(te, t, v) * np.interp(te, t, x) / dist          # speed away from the phone
        rpm_true = ra[m] * (C + vr) / C
        out[g] = (te, rpm_true / (PRIMARY * FINAL * GEARS[g]) * 2 * np.pi * R_TYRE / 60, rpm_true)
    return out

if __name__ == "__main__":
    kin = np.load("work/run2_kin.npz")
    for g, (te, va, rt) in audio_speed(kin).items():
        vv = np.interp(te, kin["t"], kin["v"])
        print(f"gear {g}: rpm {rt.min():6.0f}-{rt.max():6.0f}  audio speed {va.min():5.1f}-{va.max():5.1f} m/s  "
              f"audio/video median {np.median(va / vv):.3f}")
