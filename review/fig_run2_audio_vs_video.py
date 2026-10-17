# run2: audio spectrogram (engine orders) vs video-derived speed/accel on one time axis
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, sys
from scipy.io import wavfile; from scipy.signal import spectrogram
sys.path.insert(0, "review"); from smoother import rts_xva
d = np.load("work/run2_car_lane.npy"); d = d[(d[:, 1] >= 1.57) & (d[:, 1] <= 6.75)]
t, x = d[:, 1], d[:, 2] * 5.0
Xs, S = rts_xva(t, x, np.full(len(t), 0.08), 20)
fs, y = wavfile.read("audio/run2.wav"); y = y.astype(float) / 32768
f, ts, P = spectrogram(y[:int(8 * fs)], fs, nperseg=2048, noverlap=2048 - 110, nfft=8192)
fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1, 1]))
ax[0].pcolormesh(ts, f, 10 * np.log10(P + 1e-12), shading="auto", cmap="magma", vmin=-110, vmax=-55)
ax[0].set_ylim(250, 1300); ax[0].set_ylabel("audio freq [Hz]")
ax[0].set_title("run2: engine harmonics (audio) vs video speed — sound arrives ~0.1-0.3 s late (distance/343 m/s)")
ax[1].plot(t, Xs[:, 1], "tab:blue"); ax[1].fill_between(t, Xs[:, 1] - 2 * S[:, 1], Xs[:, 1] + 2 * S[:, 1], alpha=0.2)
ax[1].set_ylabel("vx [m/s] (video)"); ax[1].set_ylim(5, 25); ax[1].grid(alpha=0.3)
ax[2].plot(t, Xs[:, 2] / 9.81, "tab:blue"); ax[2].fill_between(t, (Xs[:, 2] - 2 * S[:, 2]) / 9.81, (Xs[:, 2] + 2 * S[:, 2]) / 9.81, alpha=0.2)
ax[2].set_ylabel("ax [g] (video)"); ax[2].set_ylim(-0.6, 1.4); ax[2].grid(alpha=0.3); ax[2].set_xlabel("video time [s]")
ax[2].set_xlim(0.5, 7.5)
plt.tight_layout(); plt.savefig("review/run2_audio_vs_video.png", dpi=90)
