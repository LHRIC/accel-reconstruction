# Dump selected full-res frames: python dump_frames.py run idx idx idx ...
import cv2, sys
run = int(sys.argv[1]); idxs = [int(a) for a in sys.argv[2:]]
cap = cv2.VideoCapture(f"videos/run{run}.mov")
i = 0
while idxs:
    ok, f = cap.read()
    if not ok: break
    if i in idxs:
        cv2.imwrite(f"frames/run{run}_{i:04d}.png", f); idxs.remove(i)
    i += 1
