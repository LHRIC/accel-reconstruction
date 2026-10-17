# Spectrograms of each run's audio (engine orders -> RPM / shifts)
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
fig, axs = plt.subplots(4, 1, figsize=(13, 13))
for k, run in enumerate([1, 2, 3, 4]):
    fs, y = wavfile.read(f"audio/run{run}.wav")
    y = y.astype(float) / 32768
    f, t, S = spectrogram(y, fs, nperseg=4096, noverlap=4096 - 441)
    ax = axs[k]
    ax.pcolormesh(t, f, 10 * np.log10(S + 1e-12), shading="auto", cmap="magma", vmin=-110, vmax=-50)
    ax.set_ylim(0, 1200); ax.set_ylabel(f"run{run}  Hz")
axs[-1].set_xlabel("video time [s]")
plt.tight_layout(); plt.savefig("review/spectrograms_raw.png", dpi=80)
