# Detect orange/red cones in the run2 panorama (color threshold -> blobs -> base point)
import cv2, numpy as np
m = cv2.imread("review/run2_mosaic.jpg")
hsv = cv2.cvtColor(m, cv2.COLOR_BGR2HSV)
mask = (((hsv[..., 0] < 12) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 90)).astype(np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
cones = []
for k in range(1, n):
    x, y, w, h, a = st[k]
    if a < 6 or a > 3000 or h < 3 or w > 3 * h:   # cones are small and taller than wide-ish
        continue
    cones.append((x + w / 2, y + h, w, h, a))    # base point = bottom center
cones = np.array(cones)
print(len(cones), "cone candidates")
np.save("work/run2_mosaic_cones.npy", cones)
vis = m.copy()
for x, y, w, h, a in cones:
    cv2.circle(vis, (int(x), int(y)), 6, (255, 255, 0), 2)
cv2.imwrite("review/run2_mosaic_cones.jpg", vis[0:1300, 1700:4381])
