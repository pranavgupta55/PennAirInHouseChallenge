# sim_fullscreen_grid.py (final)
import numpy as np, cv2, math, time

# ---------------- Config ----------------
BASE_W, BASE_H = 960, 540       # base camera render aspect [no distortion]
GRID_COLS, GRID_ROWS = 3, 3     # 3x3 fullscreen grid
MAIN_COLS, MAIN_ROWS = 2, 2     # main spans 2x2 tiles (0,0)

# ---------------- Camera / geometry ----------------
RING_DIAM_M = 0.9144
RING_RAD_M  = RING_DIAM_M / 2.0

def K_from_hfov(hfov_deg, w=BASE_W, h=BASE_H):
    hfov = math.radians(hfov_deg)
    fx = (w/2) / math.tan(hfov/2)
    fy = fx
    cx, cy = w/2, h/2
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float64), fx, fy, cx, cy  # pinhole intrinsics [web:33]

def R_from_euler_xyz(rx, ry, rz):
    rx, ry, rz = map(math.radians, (rx, ry, rz))
    Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
    Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
    Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx  # compose ZYX

def rvec_from_R(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec.astype(np.float64)

def ring_points3d(radius_m=RING_RAD_M, n=128):
    t = np.linspace(0, 2*np.pi, n, endpoint=True)
    return np.vstack([radius_m*np.cos(t), radius_m*np.sin(t), np.zeros_like(t)]).T.astype(np.float64)

def add_noise_blur(img, noise_std=0, blur_k=0):
    out = img
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if blur_k > 0 and blur_k % 2 == 1:
        out = cv2.GaussianBlur(out, (blur_k, blur_k), 0)
    return out

# ---------------- World grid cube ----------------
def cube_grid_segments(size=8.0, step=1.0):
    s = size
    z0, z1 = 0.1, s
    segs = []
    xs = np.arange(-s/2, s/2+1e-6, step)
    ys = np.arange(-s/2, s/2+1e-6, step)
    zs = np.arange(z0, z1+1e-6, step)
    for x in xs: segs.append(((x,-s/2,z1),(x,s/2,z1)))
    for y in ys: segs.append(((-s/2,y,z1),(s/2,y,z1)))
    for x in xs: segs.append(((x,-s/2,z0),(x,-s/2,z1)))
    for z in zs: segs.append(((-s/2,-s/2,z),(s/2,-s/2,z)))
    for x in xs: segs.append(((x,s/2,z0),(x,s/2,z1)))
    for z in zs: segs.append(((-s/2,s/2,z),(s/2,s/2,z)))
    for y in ys: segs.append(((-s/2,y,z0),(-s/2,y,z1)))
    for z in zs: segs.append(((-s/2,-s/2,z),(-s/2,s/2,z)))
    for y in ys: segs.append(((s/2,y,z0),(s/2,y,z1)))
    for z in zs: segs.append(((s/2,-s/2,z),(s/2,s/2,z)))
    return np.array(segs, dtype=np.float64)

def world_to_cam(Pw, R_cam, t_cam):
    # OpenCV extrinsics: X_c = R_c^T * (X_w - t_cam)
    return (np.asarray(Pw, np.float64) - t_cam) @ R_cam  # transform to camera frame [web:117]

def draw_cube_world(img, K, R_cam, t_cam, segs_world, color=(230,230,230)):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    for a_w, b_w in segs_world:
        a_c = world_to_cam(a_w, R_cam, t_cam); b_c = world_to_cam(b_w, R_cam, t_cam)
        Za, Zb = max(a_c[2], 1e-3), max(b_c[2], 1e-3)
        ua = int(round(fx * a_c[0]/Za + cx)); va = int(round(fy * a_c[1]/Za + cy))
        ub = int(round(fx * b_c[0]/Zb + cx)); vb = int(round(fy * b_c[1]/Zb + cy))
        cv2.line(img, (ua,va), (ub,vb), color, 1, cv2.LINE_AA)

# ---------------- FAST detector ----------------
def detect_ellipses_fast(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:,:,1]
    _, mask = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)     # Otsu on S [web:103]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)            # close gaps [web:104]
    band = cv2.morphologyEx(mask2, cv2.MORPH_GRADIENT, k)                       # thin rim band [web:104]
    contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ellipses = []
    cand = frame_bgr.copy()
    for c in contours:
        if len(c) < 40: 
            continue
        try: e = cv2.fitEllipseAMS(c)                                           # robust fit [web:16]
        except Exception:
            try: e = cv2.fitEllipse(c)
            except Exception: continue
        (cx,cy),(MA,ma),ang = e
        if min(MA,ma) < 12: 
            continue
        ellipses.append(e)
        cv2.ellipse(cand, e, (0,200,255), 2, cv2.LINE_AA)
    final_overlay = frame_bgr.copy()
    for e in ellipses:
        cv2.ellipse(final_overlay, e, (0,255,0), 2, cv2.LINE_AA)
    return ellipses, S, mask2, band, cand, final_overlay

# ---------------- Center/normal (approx) ----------------
def center_xyz_from_ellipse(e, fx, fy, cx, cy, radius_m=RING_RAD_M):
    (u,v), (MA,ma), _ = e
    a_px = max(MA, ma) / 2.0
    if a_px <= 1.0: return None
    z = (fx * radius_m) / a_px
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    return float(X), float(Y), float(z)

def normal_from_ellipse(e, fx, fy):
    (u,v),(MA,ma), ang = e
    a_px = max(MA,ma)/2.0; b_px = min(MA,ma)/2.0
    if a_px <= 1.0: return np.array([0,0,1], dtype=np.float64)
    tilt = np.arccos(np.clip(b_px / a_px, 0.0, 1.0))
    theta = np.deg2rad(ang)
    vmin_img = np.array([-np.sin(theta),  np.cos(theta)])
    nx = (vmin_img[0] / max(fx,1e-6)) * np.sin(tilt)
    ny = (vmin_img[1] / max(fy,1e-6)) * np.sin(tilt)
    nz = np.cos(tilt)
    n = np.array([nx, ny, nz], dtype=np.float64)
    n /= (np.linalg.norm(n) + 1e-9)
    return n

# ---------------- Panels ----------------
def label_panel(im, text, w=240):
    out = im.copy()
    cv2.rectangle(out, (6,6), (6+w, 6+22), (0,0,0), -1)
    cv2.putText(out, text, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
    return out

def render_data_panel(K, hfov, n_rings, n_detect, fps, cam_t, cam_rpy,
                      noise_std, blur_k, step_m, var_deg,
                      tx_rate, ty_rate, tz_rate, ang_rate,
                      tile_w, tile_h):
    p = np.full((tile_h, tile_w, 3), 20, np.uint8)
    x0,y0 = 10, 16
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.50; th = 1; lh = 18
    def put(line, dy=lh):
        nonlocal y0
        cv2.putText(p, line, (x0, y0), font, fs, (240,240,240), th, cv2.LINE_AA)
        y0 += dy
    put("Data")
    put(f"FPS: {fps:5.1f}   Rings: {n_rings}   Det: {n_detect}")
    put(f"FOV: {hfov:.1f} deg")
    put("K (px):")
    for r in K:
        put(f"[ {r[0]:6.1f} {r[1]:6.1f} {r[2]:6.1f} ]")
    put(f"Noise: {int(noise_std):3d}   Blur: {int(blur_k):2d}")
    put(f"Sep: {step_m:.2f} m   Var: {var_deg:.1f} deg")
    put(f"t (m): ({cam_t[0]:.2f},{cam_t[1]:.2f},{cam_t[2]:.2f})")
    rx,ry,rz = cam_rpy
    put(f"rpy (deg): ({rx:.1f},{ry:.1f},{rz:.1f})")
    put("Speeds:")
    put(f" TX/TY/TZ {tx_rate:.2f}/{ty_rate:.2f}/{tz_rate:.2f} m/s")
    put(f" ANG {ang_rate:.1f} deg/s")
    return p

def render_keybinds_panel(canvas, tile_w, tile_h):
    # compact 2-column grid overlay inside main view
    panel_w = min(tile_w*1.8 - 24, 700)
    panel_h = 22*5 + 16
    ov = canvas.copy()
    cv2.rectangle(ov, (12,12), (12+panel_w, 12+panel_h), (0,0,0), -1)
    canvas[:] = cv2.addWeighted(ov, 0.55, canvas, 0.45, 0)
    col1 = 20; col2 = panel_w//2 + 24
    y = 12+22
    items1 = [
        "Move: A/D left-right",
        "Move: W/S up-down",
        "Move: Q/E back/forward",
        "Rings: 1/2 -/+",
        "Sep: ,/. -/+",
    ]
    items2 = [
        "Pitch: I/K",
        "Yaw:   J/L",
        "Roll:  U/O",
        "Noise: N/M -/+",
        "Blur:  B/V -/+  P: occluder  H: help  Esc: quit",
    ]
    for s in items1:
        cv2.putText(canvas, s, (12+col1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA); y += 22
    y = 12+22
    for s in items2:
        cv2.putText(canvas, s, (12+col2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA); y += 22
    return canvas

# ---------------- Main ----------------
cv2.namedWindow('Sim', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Sim', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # fullscreen canvas [web:113]

# Camera pose (dark-mode default, front-right view, slight tilt)
cam_t = np.array([4.93, -3.20, -2.03], dtype=np.float64)   # x right, y down, z forward
cam_rx, cam_ry, cam_rz = -16.3, -45.6, -2.1               # deg (I/K pitch, J/L yaw, U/O roll)
hfov = 78.0

# Rings
n_rings = 4
step_m  = 0.9
var_deg = 12.0

# Effects (start with a touch of noise and blur)
noise_std = 7
blur_k = 3
occ_on = 0
help_on = True

# Movement rates
TX_RATE = 1.5; TY_RATE = 1.5; TZ_RATE = 1.6
ANG_RATE = 120.0
NOISE_RATE = 120; BLUR_RATE = 24

ring_poly = ring_points3d(n=128)
cube_segs = cube_grid_segments(size=8.0, step=1.0)

tprev = time.perf_counter(); fps = 0.0; fps_alpha = 0.1

def ring_world_pose(i):
    base_z = 3.0
    z_i = base_z + i*step_m
    t_w = np.array([0.0, 0.0, z_i], dtype=np.float64)
    rxi = var_deg * math.sin(0.7*i)
    ryi = var_deg * math.cos(0.9*i)
    rzi = var_deg * math.sin(1.3*i)
    R_w = R_from_euler_xyz(rxi, ryi, rzi)
    return R_w, t_w

def obj_to_cam(R_w, t_w, R_cam, t_cam):
    R_oc = R_cam.T @ R_w
    t_oc = R_cam.T @ (t_w - t_cam)
    return rvec_from_R(R_oc), t_oc.reshape(3,1)

ARROW_LEFT  = {81, 65361}; ARROW_RIGHT = {83, 65363}
ARROW_UP    = {82, 65362}; ARROW_DOWN  = {84, 65364}

# A small fallback canvas to show while window rects are invalid
_fallback_canvas = np.zeros((BASE_H, BASE_W, 3), np.uint8)

while True:
    tnow = time.perf_counter(); dt = max(1e-3, tnow - tprev); tprev = tnow
    fps = (1.0/dt) if fps == 0 else (1-fps_alpha)*fps + fps_alpha*(1.0/dt)

    k = cv2.waitKey(1) & 0xFFFF
    if k == 27: break

    # Translate (A/D/W/S/Q/E) in camera axes
    R_cam = R_from_euler_xyz(cam_rx, cam_ry, cam_rz)
    d = np.zeros(3, dtype=np.float64)
    if k == ord('a') or k in ARROW_LEFT:   d[0] -= TX_RATE*dt
    if k == ord('d') or k in ARROW_RIGHT:  d[0] += TX_RATE*dt
    if k == ord('w') or k in ARROW_UP:     d[1] -= TY_RATE*dt
    if k == ord('s') or k in ARROW_DOWN:   d[1] += TY_RATE*dt
    if k == ord('q'):                      d[2] += TZ_RATE*dt
    if k == ord('e'):                      d[2] -= TZ_RATE*dt
    cam_t += R_cam @ d  # move in camera frame [web:33]

    # Rotate (I/K pitch, J/L yaw, U/O roll)
    if k == ord('j'): cam_ry -= ANG_RATE*dt
    if k == ord('l'): cam_ry += ANG_RATE*dt
    if k == ord('u'): cam_rz -= ANG_RATE*dt
    if k == ord('o'): cam_rz += ANG_RATE*dt
    if k == ord('k'): cam_rx -= ANG_RATE*dt
    if k == ord('i'): cam_rx += ANG_RATE*dt

    # Rings and effects
    if k == ord('1'): n_rings = max(1, n_rings-1)
    if k == ord('2'): n_rings = min(30, n_rings+1)
    if k == ord(','): step_m  = max(0.3, step_m - 0.4*dt)
    if k == ord('.'): step_m  = min(2.0, step_m + 0.4*dt)
    if k == ord('p'): occ_on = 1 - occ_on
    if k == ord('m'): noise_std = min(255, noise_std + int(NOISE_RATE*dt))
    if k == ord('n'): noise_std = max(0,   noise_std - int(NOISE_RATE*dt))
    if k == ord('v'): blur_k = min(31, blur_k + int(BLUR_RATE*dt) | 1)
    if k == ord('b'): blur_k = max(0,  blur_k - int(BLUR_RATE*dt))
    if k == ord('h'): help_on = not help_on

    # Intrinsics and camera rotation
    K, fx, fy, cx, cy = K_from_hfov(hfov)
    R_cam = R_from_euler_xyz(cam_rx, cam_ry, cam_rz)

    # Main camera render (dark mode background)
    main = np.full((BASE_H, BASE_W, 3), 28, np.uint8)          # dark gray background
    draw_cube_world(main, K, R_cam, cam_t, cube_segs, color=(230,230,230))  # white grid
    for i in range(n_rings):
        Rw, tw = ring_world_pose(i)
        rvec, tvec = obj_to_cam(Rw, tw, R_cam, cam_t)
        pts2d, _ = cv2.projectPoints(ring_poly, rvec, tvec, K, None)           # projection [web:117]
        pts = np.round(pts2d).astype(np.int32).reshape(-1,1,2)
        col = (0, int(50 + 205*(i/max(1,n_rings-1))), 255)
        cv2.polylines(main, [pts], True, col, thickness=10, lineType=cv2.LINE_AA)
    if occ_on:
        x = int(0.45*BASE_W); y = int(0.60*BASE_H); w = int(0.18*BASE_W); h = int(0.10*BASE_H)
        cv2.rectangle(main, (x,y), (x+w, y+h), (200,50,200), thickness=-1)

    main_noisy = add_noise_blur(main, noise_std=noise_std, blur_k=blur_k if blur_k%2==1 else max(1, blur_k-1))

    # Vision
    ellipses, S, mask, band, cand_overlay, final_overlay = detect_ellipses_fast(main_noisy)
    for e in ellipses:
        xyz = center_xyz_from_ellipse(e, fx, fy, cx, cy, RING_RAD_M)
        if xyz is None: continue
        X,Y,Z = xyz
        (u,v),(MA,ma),ang = e
        n = normal_from_ellipse(e, fx, fy)
        cv2.circle(final_overlay, (int(u),int(v)), 3, (0,255,255), -1, cv2.LINE_AA)
        L = max(0.3, 0.15*Z)
        p1 = (X + L*n[0], Y + L*n[1], Z + L*n[2])
        u0 = int(round(fx*X/Z + cx)); v0 = int(round(fy*Y/Z + cy))
        u1 = int(round(fx*p1[0]/p1[2] + cx)); v1 = int(round(fy*p1[1]/p1[2] + cy))
        cv2.arrowedLine(final_overlay, (u0,v0), (u1,v1), (0,255,255), 2, tipLength=0.15, line_type=cv2.LINE_AA)

    n_detect = len(ellipses)

    # ---------------- Grid assembly (maximize height, preserve aspect) ----------------
    try:
        rect = cv2.getWindowImageRect('Sim')  # (x,y,w,h) [web:113]
        scr_w = int(rect[2]) if rect and len(rect) >= 4 and rect[2] > 0 else BASE_W*3//2
        scr_h = int(rect[3]) if rect and len(rect) >= 4 and rect[3] > 0 else BASE_H*3//2
    except Exception:
        scr_w, scr_h = BASE_W*3//2, BASE_H*3//2

    cam_aspect = float(BASE_W) / float(BASE_H)

    # Compute the largest valid tile size that fits width and height, clamp to >= 1
    tile_h_max_by_h = max(1, scr_h // GRID_ROWS)
    tile_w_if_max_h = max(1, int(tile_h_max_by_h * cam_aspect))
    if tile_w_if_max_h * GRID_COLS <= scr_w:
        tile_h = tile_h_max_by_h
        tile_w = tile_w_if_max_h
    else:
        tile_w = max(1, scr_w // GRID_COLS)
        tile_h = max(1, int(tile_w / cam_aspect))

    # Final safety clamps
    tile_w = max(1, tile_w)
    tile_h = max(1, tile_h)

    # If the sizes are still extremely small due to window race (0..7), show a safe fallback canvas once
    if tile_w < 8 or tile_h < 8:
        cv2.imshow('Sim', _fallback_canvas)
        # skip heavy resize/drawing this frame to avoid OpenCV assert and allow windowing to settle
        continue

    canvas_w = tile_w * GRID_COLS
    canvas_h = tile_h * GRID_ROWS
    canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)

    # Place main (2x2)
    main_rs = cv2.resize(main_noisy, (tile_w*2, tile_h*2), interpolation=cv2.INTER_AREA)  # preserve aspect [web:113]
    canvas[0:tile_h*2, 0:tile_w*2] = main_rs

    # Small, grid-style keybinds overlay (only once)
    if help_on:
        canvas[0:tile_h*2, 0:tile_w*2] = render_keybinds_panel(canvas[0:tile_h*2, 0:tile_w*2], tile_w, tile_h)

    # Panels
    S8 = (S * (255.0/max(1.0, S.max()))).astype(np.uint8)
    def to3(x): return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if x.ndim==2 else x
    panels = [
        label_panel(cv2.resize(to3(S8), (tile_w, tile_h), interpolation=cv2.INTER_AREA), "S channel"),
        label_panel(cv2.resize(to3(mask), (tile_w, tile_h), interpolation=cv2.INTER_AREA), "Mask"),
        label_panel(cv2.resize(to3(band), (tile_w, tile_h), interpolation=cv2.INTER_AREA), "Edge band"),
        label_panel(cv2.resize(cand_overlay, (tile_w, tile_h), interpolation=cv2.INTER_AREA), "Ellipse candidates"),
        label_panel(cv2.resize(final_overlay, (tile_w, tile_h), interpolation=cv2.INTER_AREA), "Final overlay"),
    ]

    data_panel = render_data_panel(K, hfov, n_rings, n_detect, fps, cam_t, (cam_rx,cam_ry,cam_rz),
                                   noise_std, blur_k, step_m, var_deg,
                                   TX_RATE, TY_RATE, TZ_RATE, ANG_RATE,
                                   tile_w, tile_h)

    # Slots: bottom row then right column
    slots = [(2,0), (2,1), (2,2), (0,2), (1,2)]
    for im,(r,c) in zip(panels, slots):
        y0, x0 = r*tile_h, c*tile_w
        canvas[y0:y0+tile_h, x0:x0+tile_w] = im

    # Data panel in top-right
    y0, x0 = 0, 2*tile_w
    canvas[y0:y0+tile_h, x0:x0+tile_w] = data_panel

    cv2.imshow('Sim', canvas)

# Cleanup
cv2.destroyAllWindows()
