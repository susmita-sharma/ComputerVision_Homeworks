# Optical flow summary - Video 1 (v1_1791237462462_IMG_2117.MOV)

- Source: 1080x1920 @ 30.00 fps, clip length 43.23 s
- Analysed: 29.97 s from t=0.0 s, every 1 frame(s) -> 30.00 fps, at 224x400 (899 frame pairs)
- Peak mean |flow| 0.38 px/frame at t=5.13 s; average 0.12
- Moving area: peak 11.1%, average 2.7%
- Moving objects: up to 3 at once, present in 53% of frames

## What the optical flow tells us

- When: something was moving (>0.5% of pixels moving more than 1 px/frame, or a moving object detected) in 53% of the 30.0s analysed. Peak activity at t=5.13s, when 11.1% of the frame moved; the fastest pixels then moved 3.1 px/frame (~94 px/s at analysis resolution).
- Static periods: 47% of frames are essentially still (no object, <0.5% of pixels moving), e.g. around t=23.0s.
- Camera ego-motion: essentially static (median global shift 0.03 px/frame) - so the flow that does appear comes from objects moving in the scene, not the camera.
- What moved: independent moving regions (motion left after removing camera motion) were present in 53% of frames, up to 3 at once; 8 object track(s) persisted for at least 0.5s.
- Object 1: visible t=1.63-6.57s, moving up at ~1.54 px/frame (46.2 px/s), centroid travelled 230.4 px.
- Object 5: visible t=6.83-9.67s, moving up at ~2.23 px/frame (66.8 px/s), centroid travelled 121.2 px.
- Object 6: visible t=11.9-14.67s, moving up at ~2.34 px/frame (70.1 px/s), centroid travelled 132.3 px.
- Direction: the dominant flow direction over moving frames was up-left (242 deg in image coords, 0 = right, 90 = down).
- Reliability: 29% of pixels have almost no texture (|gradient| < 2 grey levels/px). There brightness constancy gives no constraint (aperture problem), so flow in those areas is filled in by smoothing rather than measured.

## Object tracks (longest first)

| id | t start | t end | mean speed (px/frame) | direction | path (px) |
|---|---|---|---|---|---|
| 1 | 1.63 | 6.57 | 1.54 | up | 230.4 |
| 5 | 6.83 | 9.67 | 2.23 | up | 121.2 |
| 6 | 11.9 | 14.67 | 2.34 | up | 132.3 |
| 7 | 15.47 | 18.2 | 2.29 | up | 125.3 |
| 8 | 18.0 | 20.53 | 2.08 | up | 106.8 |
| 3 | 4.17 | 6.47 | 2.7 | down-left | 144.4 |
| 4 | 5.8 | 7.67 | 2.54 | down-left | 139.8 |
| 2 | 2.77 | 3.33 | 2.13 | up-left | 30.6 |

## Files

- flow_tracking.webm - 2x2 visualization video (plays at the analysed frame rate = real time)
- stats.csv - per-frame numbers; stats.png - plots
- objects.csv - every object track
- keyframe_*.png - snapshots
- pair_frame_t.png / pair_frame_t1.png - consecutive full-res frames for tracking validation
