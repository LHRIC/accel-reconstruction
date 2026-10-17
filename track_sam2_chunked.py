"""Same as track_sam2.py but in chunks (saves after each chunk, prints progress).
usage: python -u track_sam2_chunked.py <first_frame> <last_frame>"""
import numpy as np, torch, os, sys, time, shutil
from sam2.build_sam import build_sam2_video_predictor
a, b = int(sys.argv[1]), int(sys.argv[2])
prompts = {  # obj_id: {frame: box [x0, y0, x1, y1]}  (same boxes as track_sam2.py)
    1: {0: [286, 134, 364, 173], 150: [229, 141, 306, 183], 300: [187, 138, 247, 177]},
    2: {0: [285, 157, 296, 170], 150: [229, 166, 240, 178]},
    3: {0: [316, 160, 329, 174], 150: [264, 169, 277, 183]},
}
tmp = f"work/sam2_chunk_{a}_{b}"
os.makedirs(tmp, exist_ok=True)
for i in range(a, b + 1):
    shutil.copy(f"work/run2_jpg/{i:05d}.jpg", f"{tmp}/{i - a:05d}.jpg")
predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml", "models/sam2.1_hiera_small.pt", device="cuda")
masks = np.zeros((b - a + 1, 3, 360, 640), bool); t0 = time.time()
with torch.inference_mode():
    state = predictor.init_state(video_path=tmp, offload_video_to_cpu=True)
    print(f"loaded {b - a + 1} frames in {time.time() - t0:.0f}s", flush=True)
    for obj, d in prompts.items():
        for fr, box in d.items():
            if a <= fr <= b:
                predictor.add_new_points_or_box(state, frame_idx=fr - a, obj_id=obj, box=np.array(box, np.float32))
    for fr, ids, logits in predictor.propagate_in_video(state):
        for k, oid in enumerate(ids):
            masks[fr, oid - 1] = (logits[k, 0] > 0).cpu().numpy()
        if fr % 20 == 0:
            print(f"frame {fr + a}  ({time.time() - t0:.0f}s)", flush=True)
np.savez_compressed(f"work/run2_masks_{a}_{b}.npz", packed=np.packbits(masks, axis=-1), shape=np.array(masks.shape))
print("saved", flush=True)
