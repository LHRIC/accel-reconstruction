# Quick look: tile every Nth frame of each run into one image (with time stamps)
import cv2, numpy as np, sys
for run in [1, 2, 3, 4]:
    cap = cv2.VideoCapture(f"videos/run{run}.mov")
    frames, times = [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    n = len(frames)
    dt = np.diff(times)
    print(f"run{run}: {n} frames, t_end={times[-1]:.3f}s, dt mean={dt.mean()*1000:.2f}ms min={dt.min()*1000:.2f} max={dt.max()*1000:.2f}")
    np.save(f"frames/run{run}_t.npy", np.array(times))
    step = 30  # ~0.5 s
    picks = list(range(0, n, step))
    tiles = []
    for i in picks:
        im = cv2.resize(frames[i], (320, 180))
        cv2.putText(im, f"{times[i]:.1f}s #{i}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(im)
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r*cols:(r+1)*cols]) for r in range(len(tiles)//cols)]
    cv2.imwrite(f"review/run{run}_contact.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
