# Annotated run2 panorama: cone map (2 edges, indexed), lane lines -> vanishing point, car track coloured by time
import cv2, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = np.load("work/run2_lane.npz"); d = np.load("work/run2_car_raw.npy"); lane = np.load("work/run2_car_lane.npy")
def ordered(E, m, dd):
    u = (E - m) @ dd; return E[np.argsort(u * np.sign(dd[0]))]
A = ordered(L["E1"], L["m1"], L["d1"]); B = ordered(L["E2"], L["m2"], L["d2"]); V = L["V"]
X0, Y0, X1, Y1 = 2250, 150, 4381, 860
img = cv2.cvtColor(cv2.imread("review/run2_mosaic.jpg"), cv2.COLOR_BGR2RGB)[Y0:Y1, X0:X1]
fig, ax = plt.subplots(figsize=(16, 5.6)); ax.imshow(img); ax.set_axis_off()
for E, col, nm in [(A, "cyan", "A"), (B, "yellow", "B")]:
    ax.plot(E[:, 0] - X0, E[:, 1] - Y0, "o", mfc="none", mec=col, ms=9, mew=2)
    for k, p in enumerate(E):
        ax.text(p[0] - X0 + 8, p[1] - Y0 + 22, f"{nm}{k}", color=col, fontsize=7, weight="bold")
    p0 = E[0]; ax.plot([p0[0] - X0, V[0] - X0], [p0[1] - Y0, V[1] - Y0], "--", color=col, lw=1)
ok = (lane[:, 1] >= 1.57)
sc = ax.scatter(d[ok, 2] - X0, d[ok, 3] - Y0, c=d[ok, 1], cmap="viridis", s=6)
cb = plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.01); cb.set_label("video time [s]")
ax.set_xlim(0, X1 - X0); ax.set_ylim(Y1 - Y0, 0)
ax.set_title("run2: all frames stitched into one panorama (rotate+zoom camera model). Cone edges A/B -> lane lines meet at the "
             "vanishing point (off right). Dots = tracked car ground point.", fontsize=10)
plt.tight_layout(); plt.savefig("review/fig_course_panorama.png", dpi=80)
print("VP (panorama px):", V.round(0), "  cones A:", len(A), " B:", len(B))
