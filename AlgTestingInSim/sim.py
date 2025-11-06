# sim_fullscreen_grid.py (unified motion-parallax clustering + fixed graphs + ellipse overlays)
import numpy as np, cv2, math, time, os, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- Config ----------------
BASE_W, BASE_H = 960, 540
GRID_COLS, GRID_ROWS = 3, 3
MAIN_COLS, MAIN_ROWS = 2, 2

# ---------------- Camera / geometry ----------------
RING_DIAM_M = 0.9144
RING_RAD_M  = RING_DIAM_M / 2.0

def K_from_hfov(hfov_deg, w=BASE_W, h=BASE_H):
    hfov = math.radians(hfov_deg)
    fx = (w/2) / math.tan(hfov/2)
    fy = fx
    cx, cy = w/2, h/2
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float64), fx, fy, cx, cy  # [web:33]

def R_from_euler_xyz(rx, ry, rz):
    rx, ry, rz = map(math.radians, (rx, ry, rz))
    Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
    Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
    Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx

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
    return (np.asarray(Pw, np.float64) - t_cam) @ R_cam  # X_c = R^T (X_w - t_c) [web:117]

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
    _, mask = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)     # [web:104]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)            # [web:104]
    band = cv2.morphologyEx(mask2, cv2.MORPH_GRADIENT, k)                       # [web:104]
    contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ellipses = []
    cand = frame_bgr.copy()
    for c in contours:
        if len(c) < 40: 
            continue
        try: e = cv2.fitEllipseAMS(c)                                           # [web:16]
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
    put("Data"); put(f"FPS: {fps:5.1f}   Rings: {n_rings}   Det: {n_detect}")
    put(f"FOV: {hfov:.1f} deg"); put("K (px):")
    for r in K: put(f"[ {r[0]:6.1f} {r[1]:6.1f} {r[2]:6.1f} ]")
    put(f"Noise: {int(noise_std):3d}   Blur: {int(blur_k):2d}")
    put(f"Sep: {step_m:.2f} m   Var: {var_deg:.1f} deg")
    put(f"t (m): ({cam_t[0]:.2f},{cam_t[1]:.2f},{cam_t[2]:.2f})")
    rx,ry,rz = cam_rpy; put(f"rpy (deg): ({rx:.1f},{ry:.1f},{rz:.1f})")
    put("Speeds:"); put(f" TX/TY/TZ {tx_rate:.2f}/{ty_rate:.2f}/{tz_rate:.2f} m/s"); put(f" ANG {ang_rate:.1f} deg/s")
    return p

def render_keybinds_panel(canvas, tile_w, tile_h):
    panel_w = min(tile_w*1.8 - 24, 700); panel_h = 22*5 + 16
    ov = canvas.copy(); cv2.rectangle(ov, (12,12), (12+panel_w, 12+panel_h), (0,0,0), -1)
    canvas[:] = cv2.addWeighted(ov, 0.55, canvas, 0.45, 0)
    col1 = 20; col2 = panel_w//2 + 24; y = 12+22
    items1 = ["Move: A/D left-right","Move: W/S up-down","Move: Q/E back/forward","Rings: 1/2 -/+","Sep: ,/. -/+"]
    items2 = ["Pitch: I/K","Yaw:   J/L","Roll:  U/O","Noise: N/M -/+","Blur:  B/V -/+  P: occluder  T: sequence  H: help  Esc: quit"]
    for s in items1: cv2.putText(canvas, s, (12+col1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA); y += 22
    y = 12+22
    for s in items2: cv2.putText(canvas, s, (12+col2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA); y += 22
    return canvas

# ---------------- Sampling & Tracking helpers ----------------
def sample_contour_points(contour, n_pts=120):
    contour = contour.reshape(-1, 2).astype(np.float32)
    if len(contour) < 3: return np.zeros((0,2), np.float32)
    diffs = np.diff(contour, axis=0, prepend=contour[-1:])
    lengths = np.linalg.norm(diffs, axis=1)
    arc = np.cumsum(lengths); total = arc[-1] if arc.size>0 else 0.0
    if total < 1e-6: return contour[:n_pts] if len(contour)>=n_pts else contour
    sample_arcs = np.linspace(0, total, n_pts, endpoint=False)
    idxs = np.searchsorted(arc, sample_arcs); idxs = np.clip(idxs, 0, len(contour)-1)
    return contour[idxs]

def detect_mask_and_sample_points(bgr, pts_per_contour=120):
    _, S, mask, band, _, _ = detect_ellipses_fast(bgr)
    contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts_all = []; contour_ids = []
    for cid, c in enumerate(contours):
        if len(c) < 20: continue
        pts = sample_contour_points(c, n_pts=pts_per_contour)
        if pts.shape[0] == 0: continue
        pts_all.append(pts); contour_ids.extend([cid]*len(pts))
    if len(pts_all) == 0: return mask, band, np.zeros((0,2), np.float32), np.zeros((0,), np.int32)
    P = np.vstack(pts_all).astype(np.float32)
    return mask, band, P, np.array(contour_ids, np.int32)

def nn_track(prev_pts, curr_pts, max_dist=15.0):
    if len(prev_pts)==0 or len(curr_pts)==0:
        return np.full((len(prev_pts),), -1, np.int32), np.full((len(prev_pts),), np.inf, np.float32)
    D = np.linalg.norm(prev_pts[:,None,:] - curr_pts[None,:,:], axis=2)
    idxs = D.argmin(axis=1); dmin = D[np.arange(len(prev_pts)), idxs]
    idxs[dmin > max_dist] = -1
    return idxs, dmin

def next_run_dir(base="algorithmOutputVisualizations"):
    os.makedirs(base, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(base, "run_*")))
    idx = 1
    if existing:
        last = existing[-1]
        try: idx = int(os.path.basename(last).split("_")[-1]) + 1
        except: idx = len(existing) + 1
    run_dir = os.path.join(base, f"run_{idx:03d}"); os.makedirs(run_dir, exist_ok=True)
    return run_dir

def write_video(path, frames, fps=30):
    if not frames: return
    h, w = frames[0].shape[:2]; fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for fr in frames:
        if fr.shape[0] != h or fr.shape[1] != w:
            fr = cv2.resize(fr, (w,h), interpolation=cv2.INTER_AREA)
        vw.write(fr)
    vw.release()

def generate_circular_control_sequence(center_t, radius=1.2, n_frames=90):
    poses = []
    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames
        offset = np.array([0.0, radius*np.sin(theta), radius*np.cos(theta)], np.float64)
        cam_t_i = center_t + offset
        target = np.array([0.0, 0.0, 3.5], np.float64)
        fwd = target - cam_t_i; fwd /= (np.linalg.norm(fwd) + 1e-9)
        yaw = np.degrees(np.arctan2(fwd[0], fwd[2]))
        pitch = np.degrees(np.arcsin(-fwd[1])); roll = 0.0
        cam_R_i = R_from_euler_xyz(pitch, yaw, roll)
        poses.append((cam_t_i, cam_R_i))
    return poses

# ---------------- Main UI ----------------
cv2.namedWindow('Sim', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Sim', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # [web:113]

cam_t = np.array([4.93, -3.20, -2.03], np.float64)
cam_rx, cam_ry, cam_rz = -16.3, -45.6, -2.1
hfov = 78.0

n_rings = 4; step_m = 0.9; var_deg = 12.0
noise_std = 7; blur_k = 3; occ_on = 0; help_on = True

TX_RATE = 1.5; TY_RATE = 1.5; TZ_RATE = 1.6; ANG_RATE = 120.0
ring_poly = ring_points3d(n=128)
cube_segs = cube_grid_segments(size=8.0, step=1.0)

tprev = time.perf_counter(); fps = 0.0
ARROW_LEFT  = {81, 65361}; ARROW_RIGHT = {83, 65363}; ARROW_UP = {82, 65362}; ARROW_DOWN = {84, 65364}
_fallback_canvas = np.zeros((BASE_H, BASE_W, 3), np.uint8)

seq_running = False

def ring_world_pose(i):
    base_z = 3.0
    z_i = base_z + i*step_m
    t_w = np.array([0.0, 0.0, z_i], np.float64)
    rxi = var_deg * math.sin(0.7*i); ryi = var_deg * math.cos(0.9*i); rzi = var_deg * math.sin(1.3*i)
    R_w = R_from_euler_xyz(rxi, ryi, rzi)
    return R_w, t_w

def obj_to_cam(R_w, t_w, R_cam, t_cam):
    R_oc = R_cam.T @ R_w; t_oc = R_cam.T @ (t_w - t_cam)
    return rvec_from_R(R_oc), t_oc.reshape(3,1)

def render_main_frame(cam_t_loc, R_cam_loc):
    K, fx, fy, cx, cy = K_from_hfov(hfov)
    main = np.full((BASE_H, BASE_W, 3), 28, np.uint8)
    draw_cube_world(main, K, R_cam_loc, cam_t_loc, cube_segs, color=(230,230,230))
    for i in range(n_rings):
        Rw, tw = ring_world_pose(i)
        rvec, tvec = obj_to_cam(Rw, tw, R_cam_loc, cam_t_loc)
        pts2d, _ = cv2.projectPoints(ring_poly, rvec, tvec, K, None)
        pts = np.round(pts2d).astype(np.int32).reshape(-1,1,2)
        col = (0, int(50 + 205*(i/max(1,n_rings-1))), 255)
        cv2.polylines(main, [pts], True, col, thickness=10, lineType=cv2.LINE_AA)
    if occ_on:
        x = int(0.45*BASE_W); y = int(0.60*BASE_H); w = int(0.18*BASE_W); h = int(0.10*BASE_H)
        cv2.rectangle(main, (x,y), (x+w, y+h), (200,50,200), thickness=-1)
    return K, fx, fy, cx, cy, main

def track_features(tracks, W, H):
    feats = []
    for tr in tracks:
        valid = ~np.isnan(tr).any(axis=1)
        pts = tr[valid]
        if pts.shape[0] < 2:
            feats.append(np.zeros(6, np.float32)); continue
        d = np.diff(pts, axis=0)
        mags = np.linalg.norm(d, axis=1) + 1e-6
        dirs = d / mags[:,None]
        m_mean = mags.mean(); m_std = mags.std()
        u_mean = dirs.mean(axis=0)
        p_mean = (pts.mean(axis=0) / np.array([W, H], np.float32))
        feats.append(np.array([m_mean, u_mean[0], u_mean[1], p_mean[0], p_mean[1], m_std], np.float32))
    feats = np.vstack(feats)
    mu = feats.mean(0); sig = feats.std(0) + 1e-6
    feats = (feats - mu) / sig
    return feats

def kmeans_nd(features, k, iters=25):
    if features.shape[0] == 0 or k <= 1:
        return np.zeros((features.shape[0],), np.int32), features.mean(0, keepdims=True)
    # init by k-means++ style: pick first at random, rest by distance prob.
    rng = np.random.default_rng(42)
    N, D = features.shape
    centers = np.empty((k, D), np.float32)
    centers[0] = features[rng.integers(0, N)]
    d2 = np.full((N,), np.inf, np.float32)
    for c in range(1, k):
        d = np.linalg.norm(features - centers[c-1], axis=1)**2
        d2 = np.minimum(d2, d)
        probs = d2 / (d2.sum() + 1e-9)
        centers[c] = features[rng.choice(N, p=probs)]
    for _ in range(iters):
        # assign
        dist = np.linalg.norm(features[:,None,:] - centers[None,:,:], axis=2)
        labels = dist.argmin(axis=1)
        new_centers = centers.copy()
        for i in range(k):
            sel = features[labels==i]
            if len(sel) > 0:
                new_centers[i] = sel.mean(axis=0)
        if np.allclose(new_centers, centers): break
        centers = new_centers
    dist = np.linalg.norm(features[:,None,:] - centers[None,:,:], axis=2)
    labels = dist.argmin(axis=1)
    return labels, centers

def run_control_sequence():
    global seq_running
    if seq_running: return
    seq_running = True
    run_dir = next_run_dir("algorithmOutputVisualizations")
    graph_dir = os.path.join(run_dir, "graphs_frames"); os.makedirs(graph_dir, exist_ok=True)

    poses = generate_circular_control_sequence(cam_t.copy(), radius=1.2, n_frames=90)
    frames_bgr = []; frames_pts = []; stage1_frames = []

    # capture frames + sampled points
    for t_idx, (ct, Rc) in enumerate(poses):
        K, fx, fy, cx, cy, main = render_main_frame(ct, Rc)
        main_noisy = add_noise_blur(main, noise_std=noise_std, blur_k=blur_k if blur_k%2==1 else max(1, blur_k-1))
        mask, band, pts, cids = detect_mask_and_sample_points(main_noisy, pts_per_contour=120)
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for p in pts.astype(np.int32):
            cv2.circle(vis, tuple(p), 2, (0,255,0), -1)
        cv2.putText(vis, f"Frame {t_idx:03d}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        frames_bgr.append(main_noisy); frames_pts.append(pts.copy()); stage1_frames.append(vis)

    T = len(frames_pts)
    if T == 0 or len(frames_pts[0]) == 0:
        seq_running = False; return

    N0 = len(frames_pts[0])
    tracks = [np.full((T,2), np.nan, np.float32) for _ in range(N0)]
    for i in range(N0): tracks[i][0] = frames_pts[0][i]

    # NN tracking with gate
    for t_idx in range(1, T):
        prev = frames_pts[t_idx-1]; curr = frames_pts[t_idx]
        idxs, dists = nn_track(prev, curr, max_dist=20.0)
        for i, j in enumerate(idxs):
            if 0 <= j < len(curr):
                tracks[i][t_idx] = curr[j]

    # Per-frame 2D flow magnitudes + per-frame sorted graphs
    graph_frames = []
    for t_idx in range(T):
        mags_t = np.zeros((N0,), np.float32)
        if t_idx > 0:
            prev = np.array([tracks[i][t_idx-1] for i in range(N0)])
            curr = np.array([tracks[i][t_idx]   for i in range(N0)])
            valid = ~np.isnan(prev).any(axis=1) & ~np.isnan(curr).any(axis=1)
            d = curr[valid] - prev[valid]
            mags_t[valid] = np.linalg.norm(d, axis=1)  # 2D magnitude
        # sorted plot for this frame only
        y = mags_t
        mask_defined = y > 0
        ys = y[mask_defined]
        if ys.size == 0:
            ys = np.array([0.0])
        order = np.argsort(ys)
        xs_sorted = np.arange(len(ys))[order]; ys_sorted = ys[order]
        plt.figure(figsize=(8,4.5), dpi=160)
        plt.plot(xs_sorted, ys_sorted, 'o', markersize=3)
        plt.title(f"Depth proxy per point (frame {t_idx})\n(smaller is farther)")
        plt.xlabel("sorted point index"); plt.ylabel("flow magnitude (px)")
        plt.grid(True, alpha=0.3)
        png_path = os.path.join(graph_dir, f"graph_{t_idx:03d}.png")
        plt.savefig(png_path, bbox_inches='tight'); plt.close()
        img = cv2.imread(png_path); 
        if img is None: img = np.zeros((540,960,3), np.uint8)
        graph_frames.append(img)

    # Unified feature vector per track (mean mag, mean direction, mean position, mag std)
    F = track_features(tracks, BASE_W, BASE_H)
    k = max(1, min(n_rings, max(1, np.unique(~np.isnan(F).any(axis=1)).sum())))
    labels, centers = kmeans_nd(F, k=n_rings, iters=30)

    # Colored points video (cluster labels)
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,128,0),(0,128,255)]
    stage2_frames = []
    for t_idx in range(T):
        vis = cv2.cvtColor(frames_bgr[t_idx], cv2.COLOR_BGR2GRAY)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        for tid in range(N0):
            p = tracks[tid][t_idx]
            if not np.isnan(p).any():
                col = colors[labels[tid] % len(colors)]
                cv2.circle(vis, (int(p[0]), int(p[1])), 2, col, -1)
        cv2.putText(vis, f"Frame {t_idx:03d}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        stage2_frames.append(vis)

    # Fit an ellipse per cluster from all its track points; draw ellipses
    summary = frames_bgr[-1].copy()
    H, W = summary.shape[:2]
    final_ellipses = []
    for c in range(n_rings):
        tids = np.where(labels==c)[0]
        pts_all = []
        for tid in tids:
            tr = tracks[tid]
            valid = ~np.isnan(tr).any(axis=1)
            pts_all.append(tr[valid])
        if len(pts_all)==0: continue
        P = np.vstack(pts_all).astype(np.float32)
        if len(P) < 5: continue
        cnt = P.reshape(-1,1,2).astype(np.int32)
        try: e = cv2.fitEllipseAMS(cnt)
        except: 
            try: e = cv2.fitEllipse(cnt)
            except: continue
        final_ellipses.append(e)
        cv2.ellipse(summary, e, (0,255,0), 2, cv2.LINE_AA)  # draw ellipse

    # Save artifacts
    run_dir = os.path.dirname(graph_dir)
    write_video(os.path.join(run_dir, "stage1_sampled_points.mp4"), stage1_frames, fps=30)
    write_video(os.path.join(run_dir, "stage2_color_clusters.mp4"), stage2_frames, fps=30)
    write_video(os.path.join(run_dir, "stage3_depth_graphs.mp4"), graph_frames, fps=30)
    cv2.imwrite(os.path.join(run_dir, "final_object_assignments.png"), summary)

    seq_running = False

# ---------------- Main Loop ----------------
while True:
    tnow = time.perf_counter(); dt = max(1e-3, tnow - tprev); tprev = tnow
    fps = (1.0/dt) if fps == 0 else (1-0.1)*fps + 0.1*(1.0/dt)

    k = cv2.waitKey(1) & 0xFFFF
    if k == 27: break
    if k == ord('t') and not seq_running:
        run_control_sequence()

    # Translate
    R_cam = R_from_euler_xyz(cam_rx, cam_ry, cam_rz)
    d = np.zeros(3, np.float64)
    if k == ord('a') or k in ARROW_LEFT:   d[0] -= 1.5*dt
    if k == ord('d') or k in ARROW_RIGHT:  d[0] += 1.5*dt
    if k == ord('w') or k in ARROW_UP:     d[1] -= 1.5*dt
    if k == ord('s') or k in ARROW_DOWN:   d[1] += 1.5*dt
    if k == ord('q'):                      d[2] += 1.6*dt
    if k == ord('e'):                      d[2] -= 1.6*dt
    cam_t += R_cam @ d

    # Rotate
    if k == ord('j'): cam_ry -= 120.0*dt
    if k == ord('l'): cam_ry += 120.0*dt
    if k == ord('u'): cam_rz -= 120.0*dt
    if k == ord('o'): cam_rz += 120.0*dt
    if k == ord('k'): cam_rx -= 120.0*dt
    if k == ord('i'): cam_rx += 120.0*dt

    # Rings & effects
    if k == ord('1'): n_rings = max(1, n_rings-1)
    if k == ord('2'): n_rings = min(30, n_rings+1)
    if k == ord(','): step_m  = max(0.3, step_m - 0.4*dt)
    if k == ord('.'): step_m  = min(2.0, step_m + 0.4*dt)
    if k == ord('p'): occ_on = 1 - occ_on
    if k == ord('m'): noise_std = min(255, noise_std + int(120*dt))
    if k == ord('n'): noise_std = max(0,   noise_std - int(120*dt))
    if k == ord('v'): blur_k = min(31, blur_k + int(24*dt) | 1)
    if k == ord('b'): blur_k = max(0,  blur_k - int(24*dt))
    if k == ord('h'): help_on = not help_on

    # Grid tile sizing
    try:
        rect = cv2.getWindowImageRect('Sim')
        scr_w = int(rect[2]) if rect and len(rect) >= 4 and rect[2] > 0 else BASE_W*3//2
        scr_h = int(rect[3]) if rect and len(rect) >= 4 and rect[3] > 0 else BASE_H*3//2
    except Exception:
        scr_w, scr_h = BASE_W*3//2, BASE_H*3//2

    cam_aspect = float(BASE_W) / float(BASE_H)
    tile_h_max_by_h = max(1, scr_h // GRID_ROWS)
    tile_w_if_max_h = max(1, int(tile_h_max_by_h * cam_aspect))
    if tile_w_if_max_h * GRID_COLS <= scr_w:
        tile_h = tile_h_max_by_h; tile_w = tile_w_if_max_h
    else:
        tile_w = max(1, scr_w // GRID_COLS); tile_h = max(1, int(tile_w / cam_aspect))
    if tile_w < 8 or tile_h < 8:
        cv2.imshow('Sim', _fallback_canvas); continue

    canvas_w = tile_w * GRID_COLS; canvas_h = tile_h * GRID_ROWS
    canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)

    # Main render
    K, fx, fy, cx, cy = K_from_hfov(hfov)
    R_cam = R_from_euler_xyz(cam_rx, cam_ry, cam_rz)
    main = np.full((BASE_H, BASE_W, 3), 28, np.uint8)
    draw_cube_world(main, K, R_cam, cam_t, cube_segs, color=(230,230,230))
    for i in range(n_rings):
        Rw, tw = ring_world_pose(i)
        rvec, tvec = obj_to_cam(Rw, tw, R_cam, cam_t)
        pts2d, _ = cv2.projectPoints(ring_poly, rvec, tvec, K, None)
        pts = np.round(pts2d).astype(np.int32).reshape(-1,1,2)
        col = (0, int(50 + 205*(i/max(1,n_rings-1))), 255)
        cv2.polylines(main, [pts], True, col, thickness=10, lineType=cv2.LINE_AA)
    if occ_on:
        x = int(0.45*BASE_W); y = int(0.60*BASE_H); w = int(0.18*BASE_W); h = int(0.10*BASE_H)
        cv2.rectangle(main, (x,y), (x+w, y+h), (200,50,200), thickness=-1)
    main_noisy = add_noise_blur(main, noise_std=noise_std, blur_k=blur_k if blur_k%2==1 else max(1, blur_k-1))

    main_rs = cv2.resize(main_noisy, (tile_w*2, tile_h*2), interpolation=cv2.INTER_AREA)
    canvas[0:tile_h*2, 0:tile_w*2] = main_rs
    if help_on:
        canvas[0:tile_h*2, 0:tile_w*2] = render_keybinds_panel(canvas[0:tile_h*2, 0:tile_w*2], tile_w, tile_h)

    # Stages + data
    ellipses, S, mask, band, cand_overlay, final_overlay = detect_ellipses_fast(main_noisy)
    n_detect = len(ellipses)
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
    slots = [(2,0), (2,1), (2,2), (0,2), (1,2)]
    for im,(r,c) in zip(panels, slots):
        y0, x0 = r*tile_h, c*tile_w; canvas[y0:y0+tile_h, x0:x0+tile_w] = im
    y0, x0 = 0, 2*tile_w; canvas[y0:y0+tile_h, x0:x0+tile_w] = data_panel

    cv2.imshow('Sim', canvas)

# Cleanup
cv2.destroyAllWindows()
