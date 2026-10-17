"""Engine RPM from the phone audio (run2).
1) spectrogram, 2) 'comb' score = energy at engine orders 2,4,6,8 for each candidate RPM,
3) best continuous path through the score map (Viterbi): small changes cost little, big jumps cost a lot,
   except big DROPS are allowed (gear shifts).  Output: observed RPM vs audio time -> work/run2_rpm.npz"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.ndimage import median_filter

def comb_score(wav, t0, t1, orders=(2, 4, 6, 8)):
    fs, y = wavfile.read(wav); y = y.astype(float) / 32768
    f, t, S = spectrogram(y[int(t0 * fs):int(t1 * fs)], fs, window="hann", nperseg=4096, noverlap=4096 - 128, nfft=16384)
    SdB = 10 * np.log10(S + 1e-14)
    W = SdB - median_filter(SdB, size=(301, 1))             # whiten: remove broadband background
    rpm = np.arange(5000, 14501, 20.0)
    score = np.zeros((len(rpm), len(t)))
    for o in orders:
        i = np.clip(np.searchsorted(f, o * rpm / 60.0), 1, len(f) - 1)
        score += np.maximum(W[i], W[i - 1])
    return rpm, t + t0, score

def viterbi(rpm, score, max_up=60, max_drop=1200, w_smooth=0.02):
    """best path; per time step RPM may rise up to max_up, fall up to max_drop (shift), quadratic smoothness cost"""
    n_r, n_t = score.shape; step = rpm[1] - rpm[0]
    up, dn = int(max_up / step), int(max_drop / step)
    cost = score[:, 0].copy(); back = np.zeros((n_r, n_t), int)
    offs = np.arange(-up, dn + 1)                        # previous index = current + off  (off>0: rpm dropped)
    pen = w_smooth * (offs * step / 10.0) ** 2
    pen[offs > up] = w_smooth * (up * step / 10.0) ** 2 + 2.0   # flat cost for a shift-sized drop
    for k in range(1, n_t):
        best = np.full(n_r, -np.inf); arg = np.zeros(n_r, int)
        for o, p in zip(offs, pen):
            prev = np.arange(n_r) + o; ok = (prev >= 0) & (prev < n_r)
            cand = np.full(n_r, -np.inf); cand[ok] = cost[prev[ok]] - p
            better = cand > best; best[better] = cand[better]; arg[better] = prev[better]
        cost = best + score[:, k]; back[:, k] = arg
    path = np.zeros(n_t, int); path[-1] = int(np.argmax(cost))
    for k in range(n_t - 1, 0, -1): path[k - 1] = back[path[k], k]
    return rpm[path]

if __name__ == "__main__":
    rpm, t, score = comb_score("audio/run2.wav", 0.6, 6.1)
    r_obs = viterbi(rpm, score)
    np.savez("work/run2_rpm.npz", t=t, rpm_obs=r_obs, rpm_grid=rpm, score=score)
    for tt in np.arange(0.75, 6.1, 0.25): print(f"t={tt:.2f}s  observed rpm {np.interp(tt, t, r_obs):7.0f}")
