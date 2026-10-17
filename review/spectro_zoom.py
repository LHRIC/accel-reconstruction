# Zoomed spectrograms around each launch (engine order 2 = RPM/30 for an I4)
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
win = {1: (16, 25), 2: (0, 9), 3: (2, 11), 4: (11.5, 20.5)}
fig, axs = plt.subplots(4, 1, figsize=(13, 14))
for k, run in enumerate([1, 2, 3, 4]):
    fs, y = wavfile.read(f"audio/run{run}.wav")
    y = y.astype(float) / 32768
    a, b = win[run]
    seg = y[int(a * fs):int(b * fs)]
    f, t, S = spectrogram(seg, fs, nperseg=2048, noverlap=2048 - 147, nfft=8192)
    ax = axs[k]
    ax.pcolormesh(t + a, f, 10 * np.log10(S + 1e-12), shading="auto", cmap="magma", vmin=-115, vmax=-55)
    ax.set_ylim(100, 1400); ax.set_ylabel(f"run{run}  Hz")
    ax2 = ax.twinx(); ax2.set_ylim(100 * 30, 1400 * 30); ax2.set_ylabel("RPM if order 2")
axs[-1].set_xlabel("video time [s]")
plt.tight_layout(); plt.savefig("review/spectrograms_zoom.png", dpi=80)
