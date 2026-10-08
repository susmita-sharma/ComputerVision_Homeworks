# Optical flow summary - Video 2 (v2_1791237647323_IMG_2118.MOV)

- Source: 2160x3840 @ 30.00 fps, clip length 39.96 s
- Analysed: 29.96 s from t=0.0 s, every 1 frame(s) -> 30.00 fps, at 224x400 (899 frame pairs)
- Peak mean |flow| 1.42 px/frame at t=26.53 s; average 0.45
- Moving area: peak 95.8%, average 8.0%
- Moving objects: up to 5 at once, present in 76% of frames

## What the optical flow tells us

- When: something was moving (>0.5% of pixels moving more than 1 px/frame, or a moving object detected) in 77% of the 30.0s analysed. Peak activity at t=26.53s, when 95.8% of the frame moved; the fastest pixels then moved 2.0 px/frame (~60 px/s at analysis resolution).
- Static periods: 23% of frames are essentially still (no object, <0.5% of pixels moving), e.g. around t=12.6s.
- Camera ego-motion: the whole image shifted in 31% of frames (median shift (-0.04, -0.51) px/frame, image content moving up) - i.e. the camera itself was panning/moving down. This is global, smooth flow over the whole frame, unlike an object's localized flow.
- What moved: independent moving regions (motion left after removing camera motion) were present in 76% of frames, up to 5 at once; 13 object track(s) persisted for at least 0.5s.
- Object 17: visible t=15.96-26.7s, moving right at ~1.92 px/frame (57.7 px/s), centroid travelled 1098.2 px.
- Object 1: visible t=0.03-8.73s, moving right at ~2.02 px/frame (60.7 px/s), centroid travelled 912.2 px.
- Object 30: visible t=26.73-29.96s, moving down-left at ~2.09 px/frame (62.7 px/s), centroid travelled 576.2 px.
- Direction: the dominant flow direction over moving frames was right (3 deg in image coords, 0 = right, 90 = down).
- Reliability: 5% of pixels have almost no texture (|gradient| < 2 grey levels/px). There brightness constancy gives no constraint (aperture problem), so flow in those areas is filled in by smoothing rather than measured.

## Object tracks (longest first)

| id | t start | t end | mean speed (px/frame) | direction | path (px) |
|---|---|---|---|---|---|
| 17 | 15.96 | 26.7 | 1.92 | right | 1098.2 |
| 1 | 0.03 | 8.73 | 2.02 | right | 912.2 |
| 30 | 26.73 | 29.96 | 2.09 | down-left | 576.2 |
| 24 | 23.8 | 26.36 | 1.34 | down | 424.1 |
| 19 | 19.86 | 22.3 | 1.14 | right | 196.1 |
| 22 | 22.83 | 24.23 | 1.11 | right | 221.6 |
| 23 | 23.6 | 25.0 | 1.18 | right | 185.6 |
| 38 | 28.7 | 29.96 | 1.31 | left | 199.2 |
| 25 | 23.8 | 24.96 | 1.13 | right | 103.7 |
| 37 | 28.3 | 29.4 | 1.34 | left | 131.1 |
| 5 | 1.23 | 2.03 | 1.51 | up-right | 88.3 |
| 2 | 0.23 | 0.8 | 1.65 | right | 61.1 |

## Files

- flow_tracking.webm - 2x2 visualization video (plays at the analysed frame rate = real time)
- stats.csv - per-frame numbers; stats.png - plots
- objects.csv - every object track
- keyframe_*.png - snapshots
- pair_frame_t.png / pair_frame_t1.png - consecutive full-res frames for tracking validation
