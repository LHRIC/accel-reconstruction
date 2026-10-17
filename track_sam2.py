"""Segment the car and its two rear tyres in every frame with SAM2 (video mode).
Prompts are boxes drawn (by eye) around the car / tyres in a few frames. Output: work/run2_masks.npz"""
import numpy as np, torch, os, time
from sam2.build_sam import build_sam2_video_predictor
FRAMES = "work/run2_jpg_run"
prompts = {  # obj_id: {frame: box [x0, y0, x1, y1]}
    1: {0: [286, 134, 364, 173], 150: [229, 141, 306, 183], 300: [187, 138, 247, 177]},   # whole car
    2: {0: [285, 157, 296, 170], 150: [229, 166, 240, 178]},                               # far-side rear tyre
    3: {0: [316, 160, 329, 174], 150: [264, 169, 277, 183]},                               # near-side rear tyre
}
predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml", "models/sam2.1_hiera_small.pt", device="cuda")
n = len(os.listdir(FRAMES)); masks = np.zeros((n, 3, 360, 640), bool); t0 = time.time()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    state = predictor.init_state(video_path=FRAMES, offload_video_to_cpu=True, offload_state_to_cpu=True)
    for obj, d in prompts.items():
        for fr, box in d.items():
            predictor.add_new_points_or_box(state, frame_idx=fr, obj_id=obj, box=np.array(box, np.float32))
    for fr, ids, logits in predictor.propagate_in_video(state):
        for k, oid in enumerate(ids):
            masks[fr, oid - 1] = (logits[k, 0] > 0).cpu().numpy()
        if fr % 50 == 0: print(f"frame {fr}  ({time.time()-t0:.0f}s)", flush=True)
np.savez_compressed("work/run2_masks.npz", packed=np.packbits(masks, axis=-1), shape=np.array(masks.shape))
area = masks.sum((2, 3)); print("mask area per object (min/median/max):", [(a.min(), int(np.median(a)), a.max()) for a in area.T])
print("frames with empty car mask:", np.flatnonzero(area[:, 0] == 0)[:30])
