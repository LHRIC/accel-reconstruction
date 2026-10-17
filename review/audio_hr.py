import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.io import wavfile; from scipy.signal import spectrogram
fs, y = wavfile.read("audio/run2.wav"); y = y.astype(float) / 32768
a, b = 0.5, 7.8
f, t, S = spectrogram(y[int(a * fs):int(b * fs)], fs, window="hann", nperseg=4096, noverlap=4096 - 128, nfft=16384)
t = t + a; SdB = 10 * np.log10(S + 1e-14)
np.savez("work/run2_spec.npz", f=f, t=t, SdB=SdB)
fig, ax = plt.subplots(2, 1, figsize=(16, 11))
for k, (lo, hi) in enumerate([(150, 700), (600, 1500)]):
    m = (f >= lo) & (f <= hi)
    ax[k].pcolormesh(t, f[m], SdB[m], shading="auto", cmap="magma", vmin=np.percentile(SdB[m], 40), vmax=np.percentile(SdB[m], 99.7))
    ax[k].set_ylabel("Hz"); ax[k].set_yticks(np.arange(lo, hi + 1, 50 if hi < 800 else 100)); ax[k].grid(alpha=0.2)
    ax[k].set_xticks(np.arange(0.5, 7.81, 0.25))
ax[1].set_xlabel("video time [s]"); plt.tight_layout(); plt.savefig("review/run2_spec_hr.png", dpi=75)
