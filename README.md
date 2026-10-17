# accel-reconstruction

Reconstruct FSAE acceleration-event x(t), vx(t), ax(t) from phone videos taken in the
stands at MIS (FSAE Michigan). Handheld camera that pans and zooms; cones are the
position anchors. **Run2 is done ("full send"); runs 1, 3, 4 wait until run2 is verified.**

## Runs

| label | original file (Discord attachment)        | length | official time (timing board) |
|-------|-------------------------------------------|--------|------------------------------|
| run1  | 5800C948-D42E-48F3-AD4A-3C6EC0FBEB3F.mov  | 29.6 s | 5.099 s |
| run2  | 5D3F8377-FCB5-49A4-951B-DF749EB0ED3B.mov  | 15.4 s | 4.837 s |
| run3  | CA620E5F-C467-4105-836B-52CEAF1873DB.mov  | 14.6 s | 4.801 s |
| run4  | DAFCA5F4-C820-4B3F-8CB7-08FD3D08BA0A_1.mov| 21.3 s | 4.893 s |

All 640x360, ~59 fps (variable frame rate, so use frame timestamps, not frame count / fps).
Same car, different drivers.

## How run2 works (run the scripts in this order)

1. `review/register_kf.py 2 420` - every frame's 3x3 homography to frame 0, matched against
   keyframes (the camera only turns + zooms, so one homography per frame is exact; ~0.3 px).
   Frame-to-frame chaining drifts 5-15 px - don't use `review/camera_motion.py`.
2. `review/mosaic.py`, `review/cone_map.py`, `review/lane_fit.py`, `review/vp_lines.py` - panorama,
   static cone map, lane vanishing point. `review/selfcal_fine.py` - focal length from the rotation
   (f = 2362 px in reference frame 360).
3. `calib3d.py` (+ `run2_setup.py`) - 3D ground plane: pixel -> metres along/across the lane.
   - Only the plane **tilt** matters. It is fixed with the **painted dashes** (machine painted, evenly
     spaced): `V_SHIFT = -20 px`. Roll and focal length cancel out once the scale is set (checked).
   - **Scale = 75 m from the start line to the finish line.** The start line is 0.3 m in front of the
     staged front wheels (rules). The finish is the **doubled cones**. The painted white lines at the
     green light (~10 m) and the checkered sign (~100 m) are *not* the timing lines. With them the run
     would take 4.40 s.
   - Cone spacing is NOT used for distance (human placed, gaps scatter +-10-15%).
4. `track_sam2.py` - SAM2 (video segmentation) of the car + both rear tyres (box prompts by eye in
   frames 0/150/300). Needs `pip install git+https://github.com/facebookresearch/sam2.git` and
   `models/sam2.1_hiera_small.pt`. ~30 min on this laptop GPU.
5. `kinematics.py` - near-side ("visible") rear tyre centre at 0.198 m hub height -> lane x.
   `cone_anchor.py` removes the local registration error with the cones near the car. RTS smoother
   (constant-accel model, jerk noise q = 10) -> x, v, a with +-2 sigma. Detects shifts from the accel dips.
6. `audio_rpm.py` - engine RPM from the audio (orders 2/4/6/8 'comb' + Viterbi path).
   `audio_speed.py` - speed from RPM x gearing (primary 2.111, final 2.84, CBR600 gears, r = 0.198 m),
   with the Doppler shift and sound delay removed. Used only as an independent check, never fused.
7. `plots.py` (raw + filtered windows), `make_videos.py` (overlay videos), `compare_cone_methods.py`.

Outputs are in `out/`: `run2_raw.png`, `run2_filtered.png`, `run2_accel.csv`, `run2_telemetry.mp4`,
`run2_steps.mp4`, `run2_cone_methods.mp4`, `run2_gear_ratio_check.png`, `run2_tilt_options.png`.

## Run2 checks

- Timed section start line -> finish from the video: 4.903 s vs board 4.837 s (+66 ms).
- Audio speed / video speed: 2nd gear 1.007, 3rd gear 1.001 (independent: engine sound + gearing).
- Effective gear ratio (audio rpm / video wheel rpm) sits exactly on 2.00 and 1.667 in 2nd/3rd.
- The two rear tyres tracked separately give the same v(t), a(t).

## Open questions (run2)

- In the first ~2.4 s the engine never locks to the 1st-gear ratio (2.75). The effective ratio falls
  from >10 (clutch slipping on a ~9-10k rpm hold) to ~2.3-2.5, then the first RPM drop takes it to 2.00.
  Launch in 2nd? Or a long clutch slip through an early 1->2 shift (video accel dip at ~0.95 s)?
- Video accel after shifts overshoots the audio-derived accel by up to ~0.3-0.4 g (registration/zoom).
  Audio accel is the better in-gear estimate.

Needs: `pip install opencv-contrib-python imageio-ffmpeg scipy matplotlib` (+ torch with CUDA, sam2 for step 4).
