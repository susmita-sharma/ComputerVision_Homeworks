# Structure from motion - planar object from 4 viewpoints

## Object and images

- Object: flat rectangle, measured 340.0 mm x 210.0 mm
- World frame: origin at the top-left corner, X along the top edge, Y down the side edge, Z = 0 on the object
- Image sizes (as displayed, after the phone's EXIF rotation): view 1 3024x4032, view 2 3024x4032, view 3 3024x4032, view 4 4032x3024
- Images: view1_IMG_2123.JPG, view2_IMG_2124.JPG, view3_IMG_2125.JPG, view4_IMG_2126.JPG

## Camera intrinsics (shared by all 4 views)

Source: photo EXIF: 35mm-equivalent focal length 26 mm -> f = 26/43.27 x image diagonal = 3028.4 px; principal point at the image centre. The phone saved these photos in different orientations (view 1: portrait, view 2: portrait, view 3: portrait, view 4: landscape), so each view uses the same focal length with its principal point at its own image centre

    K =
    [   3028.43       0.00    1512.00 ]
    [      0.00    3028.43    2016.00 ]
    [      0.00       0.00       1.00 ]

- fx = 3028.43 px, fy = 3028.43 px, principal point (1512.0, 2016.0)
- FOV across the long side = 2 atan(long / 2f) = 67.3 deg, across the short side = 53.1 deg
- lens distortion: assumed zero (not calibrated)

Per-view intrinsics (same focal length; axes/principal point follow each photo's orientation):

    K1 =
    [   3028.43       0.00    1512.00 ]
    [      0.00    3028.43    2016.00 ]
    [      0.00       0.00       1.00 ]

    K2 =
    [   3028.43       0.00    1512.00 ]
    [      0.00    3028.43    2016.00 ]
    [      0.00       0.00       1.00 ]

    K3 =
    [   3028.43       0.00    1512.00 ]
    [      0.00    3028.43    2016.00 ]
    [      0.00       0.00       1.00 ]

    K4 =
    [   3028.43       0.00    2016.00 ]
    [      0.00    3028.43    1512.00 ]
    [      0.00       0.00       1.00 ]

## Camera positions (extrinsics)

| cam | position C = -R^T t (mm) | distance to object centre (mm) | tilt from plane normal (deg) | Rodrigues r (deg) | reproj. RMS (px) |
|---|---|---|---|---|---|
| 1 | (454.9, 92.5, -198.0) | 347.2 | 46.2 | (6.1, 45.8, 1.3) | 562.24 |
| 2 | (-70.4, 152.2, -214.3) | 325.5 | 42.1 | (-10.0, -40.9, 0.3) | 426.09 |
| 3 | (416.2, 212.5, -292.7) | 397.3 | 37.5 | (-11.1, 35.9, 6.5) | 316.53 |
| 4 | (144.5, 383.2, -218.2) | 354.4 | 47.0 | (-46.8, -3.7, -1.4) | 285.83 |

## Step 1 - homography from the 4 corners (per view)

World corners (X, Y) = (0,0), (W,0), (W,H), (0,H). Each correspondence gives two rows of A h = 0:

    [-X -Y -1  0  0  0  xX  xY  x]
    [ 0  0  0 -X -Y -1  yX  yY  y]

h = right singular vector of A (8 x 9) with the smallest singular value; H normalised so H33 = 1.

### View 1

Clicked corners (px): (416.5, 447.1), (2330.6, 414.5), (2298.0, 3143.1), (538.7, 3175.7)

    A =
    [       -0.0        -0.0        -1.0         0.0         0.0         0.0         0.0         0.0       416.5 ]
    [        0.0         0.0         0.0        -0.0        -0.0        -1.0         0.0         0.0       447.1 ]
    [     -340.0        -0.0        -1.0         0.0         0.0         0.0    792400.9         0.0      2330.6 ]
    [        0.0         0.0         0.0      -340.0        -0.0        -1.0    140933.7         0.0       414.5 ]
    [     -340.0      -210.0        -1.0         0.0         0.0         0.0    781323.7    482582.3      2298.0 ]
    [        0.0         0.0         0.0      -340.0      -210.0        -1.0   1068664.4    660057.4      3143.1 ]
    [       -0.0      -210.0        -1.0         0.0         0.0         0.0         0.0    113123.9       538.7 ]
    [        0.0         0.0         0.0        -0.0      -210.0        -1.0         0.0    666899.3      3175.7 ]

    H =
    [     5.63685      0.80748    416.50991 ]
    [    -0.09454     14.32393    447.09145 ]
    [     0.00000      0.00042      1.00000 ]

Step 2 - pose from H:  B = K^-1 H = [b1 b2 b3],  lambda = 1/||b1||

    B =
    [    0.001860     0.000057    -0.361736 ]
    [   -0.000033     0.004451    -0.518061 ]
    [    0.000003     0.000419     1.000000 ]

    lambda = 537.6138
    r1 = lambda b1 = [0.9998, -0.0179, 0.0017]
    r2 = lambda b2 = [0.0309, 2.3929, 0.2252]
    r3 = r1 x r2  = [-0.008, -0.2252, 2.393]
    t  = lambda b3 = [-194.47, -278.52, 537.61] mm

    R (orthonormalised via SVD) =
    [   0.9999    0.0143   -0.0033 ]
    [  -0.0145    0.9955   -0.0937 ]
    [   0.0020    0.0937    0.9956 ]

Refined with solvePnP (iterative, minimises reprojection error):

    R =
    [   0.6970    0.0196    0.7168 ]
    [   0.0605    0.9945   -0.0861 ]
    [  -0.7145    0.1033    0.6919 ]
    t = [-176.93, -136.57, 452.49] mm
    difference homography-pose vs PnP: rotation 46.006 deg, translation 166.44 mm
    camera centre C = -R^T t = [454.89, 92.52, -198.03] mm

    P = K [R | t] =
    [    1030.488      215.742     3216.995   148351.610 ]
    [   -1257.260     3219.975     1134.319   498622.479 ]
    [      -0.715        0.103        0.692      452.488 ]

### View 2

Clicked corners (px): (500.0, 634.4), (2259.3, 650.7), (2821.3, 3411.9), (141.6, 3485.2)

    A =
    [       -0.0        -0.0        -1.0         0.0         0.0         0.0         0.0         0.0       500.0 ]
    [        0.0         0.0         0.0        -0.0        -0.0        -1.0         0.0         0.0       634.4 ]
    [     -340.0        -0.0        -1.0         0.0         0.0         0.0    768158.7         0.0      2259.3 ]
    [        0.0         0.0         0.0      -340.0        -0.0        -1.0    221244.7         0.0       650.7 ]
    [     -340.0      -210.0        -1.0         0.0         0.0         0.0    959241.0    592472.4      2821.3 ]
    [        0.0         0.0         0.0      -340.0      -210.0        -1.0   1160052.8    716503.2      3411.9 ]
    [       -0.0      -210.0        -1.0         0.0         0.0         0.0         0.0     29732.5       141.6 ]
    [        0.0         0.0         0.0        -0.0      -210.0        -1.0         0.0    731897.5      3485.2 ]

    H =
    [     5.32865     -1.94142    499.96461 ]
    [     0.09232      7.79421    634.42951 ]
    [     0.00007     -0.00166      1.00000 ]

Step 2 - pose from H:  B = K^-1 H = [b1 b2 b3],  lambda = 1/||b1||

    B =
    [    0.001725     0.000187    -0.334179 ]
    [   -0.000015     0.003678    -0.456201 ]
    [    0.000068    -0.001659     1.000000 ]

    lambda = 579.0755
    r1 = lambda b1 = [0.9992, -0.0087, 0.0395]
    r2 = lambda b2 = [0.1083, 2.1298, -0.9605]
    r3 = r1 x r2  = [-0.0758, 0.964, 2.129]
    t  = lambda b3 = [-193.51, -264.17, 579.08] mm

    R (orthonormalised via SVD) =
    [   0.9987    0.0397   -0.0324 ]
    [  -0.0228    0.9108    0.4123 ]
    [   0.0459   -0.4110    0.9105 ]

Refined with solvePnP (iterative, minimises reprojection error):

    R =
    [   0.7560    0.0556   -0.6522 ]
    [   0.0639    0.9854    0.1580 ]
    [   0.6514   -0.1611    0.7414 ]
    t = [-94.96, -111.67, 229.24] mm
    difference homography-pose vs PnP: rotation 40.946 deg, translation 394.15 mm
    camera centre C = -R^T t = [-70.41, 152.24, -214.25] mm

    P = K [R | t] =
    [    3274.503      -75.291     -854.090    59044.109 ]
    [    1506.676     2659.413     1973.086   123971.997 ]
    [       0.651       -0.161        0.741      229.243 ]

### View 3

Clicked corners (px): (1100.7, 600.1), (2583.1, 583.8), (2819.3, 2685.2), (302.5, 2563.0)

    A =
    [       -0.0        -0.0        -1.0         0.0         0.0         0.0         0.0         0.0      1100.7 ]
    [        0.0         0.0         0.0        -0.0        -0.0        -1.0         0.0         0.0       600.1 ]
    [     -340.0        -0.0        -1.0         0.0         0.0         0.0    878249.5         0.0      2583.1 ]
    [        0.0         0.0         0.0      -340.0        -0.0        -1.0    198484.2         0.0       583.8 ]
    [     -340.0      -210.0        -1.0         0.0         0.0         0.0    958559.5    592051.4      2819.3 ]
    [        0.0         0.0         0.0      -340.0      -210.0        -1.0    912975.2    563896.5      2685.2 ]
    [       -0.0      -210.0        -1.0         0.0         0.0         0.0         0.0     63520.7       302.5 ]
    [        0.0         0.0         0.0        -0.0      -210.0        -1.0         0.0    538239.3      2563.0 ]

    H =
    [     4.03918     -4.38730   1100.69209 ]
    [    -0.12041      4.37959    600.06723 ]
    [    -0.00012     -0.00194      1.00000 ]

Step 2 - pose from H:  B = K^-1 H = [b1 b2 b3],  lambda = 1/||b1||

    B =
    [    0.001396    -0.000481    -0.135816 ]
    [    0.000043     0.002736    -0.467547 ]
    [   -0.000124    -0.001938     1.000000 ]

    lambda = 713.3008
    r1 = lambda b1 = [0.9956, 0.0306, -0.0886]
    r2 = lambda b2 = [-0.3431, 1.9519, -1.3826]
    r3 = r1 x r2  = [0.1306, 1.4069, 1.9538]
    t  = lambda b3 = [-96.88, -333.5, 713.3] mm

    R (orthonormalised via SVD) =
    [   0.9909   -0.1229    0.0542 ]
    [   0.0684    0.8092    0.5835 ]
    [  -0.1155   -0.5745    0.8103 ]

Refined with solvePnP (iterative, minimises reprojection error):

    R =
    [   0.8052   -0.1630    0.5701 ]
    [   0.0460    0.9758    0.2140 ]
    [  -0.5912   -0.1461    0.7932 ]
    t = [-133.61, -163.89, 509.29] mm
    difference homography-pose vs PnP: rotation 38.675 deg, translation 267.84 mm
    camera centre C = -R^T t = [416.2, 212.53, -292.74] mm

    P = K [R | t] =
    [    1544.756     -714.392     2925.899   365416.411 ]
    [   -1052.473     2660.565     2247.068   530391.189 ]
    [      -0.591       -0.146        0.793      509.290 ]

### View 4

Clicked corners (px): (775.2, 767.5), (3131.8, 789.3), (3881.2, 1962.2), (156.2, 1962.2)

    A =
    [       -0.0        -0.0        -1.0         0.0         0.0         0.0         0.0         0.0       775.2 ]
    [        0.0         0.0         0.0        -0.0        -0.0        -1.0         0.0         0.0       767.5 ]
    [     -340.0        -0.0        -1.0         0.0         0.0         0.0   1064828.1         0.0      3131.8 ]
    [        0.0         0.0         0.0      -340.0        -0.0        -1.0    268348.6         0.0       789.3 ]
    [     -340.0      -210.0        -1.0         0.0         0.0         0.0   1319604.5    815049.8      3881.2 ]
    [        0.0         0.0         0.0      -340.0      -210.0        -1.0    667150.1    412063.3      1962.2 ]
    [       -0.0      -210.0        -1.0         0.0         0.0         0.0         0.0     32801.5       156.2 ]
    [        0.0         0.0         0.0        -0.0      -210.0        -1.0         0.0    412063.3      1962.2 ]

    H =
    [     7.10184     -3.22373    775.21982 ]
    [     0.10687      2.22165    767.53939 ]
    [     0.00005     -0.00177      1.00000 ]

Step 2 - pose from H:  B = K^-1 H = [b1 b2 b3],  lambda = 1/||b1||

    B =
    [    0.002309     0.000112    -0.409711 ]
    [    0.000008     0.001616    -0.245824 ]
    [    0.000054    -0.001767     1.000000 ]

    lambda = 433.0021
    r1 = lambda b1 = [0.9997, 0.0035, 0.0236]
    r2 = lambda b2 = [0.0484, 0.6997, -0.7651]
    r3 = r1 x r2  = [-0.0192, 0.766, 0.6993]
    t  = lambda b3 = [-177.41, -106.44, 433.0] mm

    R (orthonormalised via SVD) =
    [   0.9993    0.0311   -0.0185 ]
    [  -0.0073    0.6743    0.7384 ]
    [   0.0355   -0.7378    0.6741 ]

Refined with solvePnP (iterative, minimises reprojection error):

    R =
    [   0.9977    0.0469   -0.0485 ]
    [   0.0033    0.6843    0.7292 ]
    [   0.0674   -0.7277    0.6826 ]
    t = [-172.7, -103.59, 418.0] mm
    difference homography-pose vs PnP: rotation 2.085 deg, translation 15.98 mm
    camera centre C = -R^T t = [144.46, 383.17, -218.16] mm

    P = K [R | t] =
    [    3157.445    -1324.957     1229.086   319687.265 ]
    [     111.902      972.004     3240.401   318314.009 ]
    [       0.067       -0.728        0.683      418.000 ]

## Step 3 - triangulating boundary point B1 (worked)

Observations: view 1 (481.7, 1880.6), view 2 (500.0, 1701.4), view 3 (750.5, 1375.9), view 4 (482.0, 1204.7)

Each view adds rows  x P3 - P1  and  y P3 - P2  (Pk = k-th row of P):

    A =
    [   -1374.646     -165.961    -2883.704    69598.371 ]
    [     -86.474    -3025.612      166.985   352342.374 ]
    [   -2948.812       -5.241     1224.770    55569.249 ]
    [    -398.307    -2933.474     -711.613   266071.564 ]
    [   -1988.400      604.783    -2330.626    16783.396 ]
    [     239.092    -2861.524    -1155.691   170336.577 ]
    [   -3124.946      974.203     -900.090  -118211.529 ]
    [     -30.678    -1848.656    -2418.127   185242.231 ]

    singular values: [529609.59673, 5587.05426, 3887.31988, 1530.26618]
    X_h (last right singular vector) = [0.161585, 0.986533, -0.023405, 0.00977]
    X = X_h[:3] / X_h[3] = [16.54, 100.98, -2.4] mm

## Reconstructed points (world mm)

| point | X | Y | Z |
|---|---|---|---|
| corner 1 | 0.41 | -120.15 | 73.153 |
| corner 2 | 316.68 | -39.83 | 14.623 |
| corner 3 | 336.68 | 231.67 | -4.482 |
| corner 4 | 24.37 | 258.08 | -19.165 |
| B1 | 16.54 | 100.98 | -2.396 |
| B2 | 174.94 | -33.75 | -3.296 |
| B3 | 336.50 | 76.53 | -5.855 |
| B4 | 181.62 | 236.60 | 71.573 |

## Estimated boundary and validation

- Edge lengths from re-triangulated corners: top 331.5, right 272.9, bottom 313.8, left 390.1 mm (measured 340.0 x 210.0)
- Estimated outline (corners + boundary points ordered around their centroid): area 94311 mm^2 (rectangle 71400 mm^2), perimeter 1313.1 mm
- Corner recovery error (mm): [140.664, 48.415, 22.38, 57.21]
- Planarity RMS (distance of all points to best-fit plane): 31.771 mm
- Reprojection RMS per view (px): [562.24, 426.09, 316.53, 285.83]
