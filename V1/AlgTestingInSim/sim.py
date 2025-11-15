# sim_fullscreen_grid.py (LK+FB tracking, stable PCA, larger baseline, optional triangulation, spatial post-split)
import numpy as np, cv2, math, time, os, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Optional Plotly for interactive 3D PCA export (best-effort; falls back if missing)
try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# Optional DBSCAN for spatial post-split
try:
    from sklearn.cluster import DBSCAN
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# ---------------- Config ----------------
BASE_W, BASE_H = 960, 540
GRID_COLS, GRID_ROWS = 3, 3
MAIN_COLS, MAIN_ROWS = 2, 2

# Feature/triangulation toggles
USE_TRIANGULATION = True          # Enable two-view triangulation depth clustering if possible
TRI_BASELINE_FRAMES = None        # None => auto: opposite frames (0, T//2)
DBSCAN_EPS = 30.0                 # pixels at final frame for spatial post-split
DBSCAN_MIN_SAMPLES = 15           # min cluster size for spatial post-split

# ---------------- Camera / geometry ----------------
RING_DIAM_M = 0.9144
RING_RAD_M  = RING_DIAM_M / 2.0

def K_from_hfov(hfov_deg, w=BASE_W, h=BASE_H):
    hfov = math.radians(hfov_deg)
    fx = (w/2) / math.tan(hfov/2)
    fy = fx
    cx, cy = w/2, h/2
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float64), fx, fy, cx, cy

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
    return (np.asarray(Pw, np.float64) - t_cam) @ R_cam  # X_c = R^T (X_w - t_c)

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
    _, mask = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)     # Otsu on S
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)            # close gaps
    band = cv2.morphologyEx(mask2, cv2.MORPH_GRADIENT, k)                       # thin rim band
    contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ellipses = []
    cand = frame_bgr.copy()
    for c in contours:
        if len(c) < 40: continue
        try: e = cv2.fitEllipseAMS(c)
        except Exception:
            try: e = cv2.fitEllipse(c)
            except Exception: continue
        (cx,cy),(MA,ma),ang = e
        if min(MA,ma) < 12: continue
        ellipses.append(e)
        cv2.ellipse(cand, e, (0,200,255), 2, cv2.LINE_AA)
    final_overlay = frame_bgr.copy()
    for e in ellipses:
        cv2.ellipse(final_overlay, e, (0,255,0), 2, cv2.LINE_AA)
    return ellipses, S, mask2, band, cand, final_overlay

# ---------------- PCA utilities (NumPy SVD) ----------------
def pca_project(X, n_components=2):
    # X: NxD
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((0, n_components), np.float32), np.zeros((X.shape[1],), np.float32), np.eye(n_components, X.shape[1], dtype=np.float32)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:n_components]          # components (PCs)
    Z = Xc @ W.T                   # projection
    return Z.astype(np.float32), mu.squeeze().astype(np.float32), W.astype(np.float32)

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
        cv2.putText(p, line, (x0, y0), font, fs, (240,240,240), th, cv2.LINE_AA); y0 += dy
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

def write_video(path, frames, fps=30):
    if not frames: return
    h, w = frames[0].shape[:2]; fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for fr in frames:
        if fr.shape[0] != h or fr.shape[1] != w:
            fr = cv2.resize(fr, (w,h), interpolation=cv2.INTER_AREA)
        vw.write(fr)
    vw.release()

# New: LK optical flow tracking with forward-backward consistency
def compute_tracks_lk(frames_bgr, init_points, fb_thresh=1.5):
    T = len(frames_bgr)
    N0 = len(init_points)
    tracks = [np.full((T,2), np.nan, np.float32) for _ in range(N0)]
    if T == 0 or N0 == 0:
        return tracks

    # Initialize
    for i in range(N0):
        tracks[i][0] = init_points[i]

    I_prev = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    p_prev = init_points.reshape(-1,1,2).astype(np.float32)

    lk_params = dict(winSize=(21,21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 1e-3))

    for t in range(1, T):
        I_curr = cv2.cvtColor(frames_bgr[t], cv2.COLOR_BGR2GRAY)
        p_next, st, err = cv2.calcOpticalFlowPyrLK(I_prev, I_curr, p_prev, None, **lk_params)

        # Backward check
        p_back, st2, _ = cv2.calcOpticalFlowPyrLK(I_curr, I_prev, p_next, None, **lk_params)
        fb_err = np.linalg.norm(p_prev - p_back, axis=2).reshape(-1)
        ok = (st.reshape(-1) > 0) & (st2.reshape(-1) > 0) & (fb_err < fb_thresh)

        for i in range(N0):
            if ok[i] and not np.any(np.isnan(p_next[i,0])):
                tracks[i][t] = p_next[i,0]
            else:
                tracks[i][t] = np.nan

        I_prev = I_curr
        p_prev = p_next

    return tracks

# ---------------- Feature builders ----------------
def per_frame_features(tracks, t, W, H, mag_hist):
    """
    Build per-frame features for PCA plots:
      f_t = [dx, dy, x_norm, y_norm, std_mag_up_to_t]
    """
    N = len(tracks)
    F = np.zeros((N, 5), np.float32)
    valid = np.zeros((N,), np.bool_)
    for i in range(N):
        tr = tracks[i]
        if t == 0: 
            p = tr[t]
            if not np.isnan(p).any():
                F[i,2:4] = p / np.array([W, H], np.float32)  # anchor
                F[i,4] = 0.0
                valid[i] = True
            continue
        p0 = tr[t-1]; p1 = tr[t]
        if not (np.isnan(p0).any() or np.isnan(p1).any()):
            dp = p1 - p0
            F[i,0:2] = dp
            F[i,2:4] = p1 / np.array([W, H], np.float32)
            hist = mag_hist[i][:t]  # mags up to t-1
            F[i,4] = hist.std() if hist.size > 1 else 0.0
            valid[i] = True
    return F, valid

def track_features_sequence(tracks, W, H):
    """
    6D unified features per track over the whole sequence:
      [mean_mag, u_mean_x, u_mean_y, mean_x/W, mean_y/H, std_mag]
    """
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

def kmeans_nd(features, k, iters=25, seed=42):
    if features.shape[0] == 0 or k <= 1:
        return np.zeros((features.shape[0],), np.int32), features.mean(0, keepdims=True)
    rng = np.random.default_rng(seed)
    N, D = features.shape
    centers = np.empty((k, D), np.float32)
    centers[0] = features[rng.integers(0, N)]
    d2 = np.full((N,), np.inf, np.float32)
    for c in range(1, k):
        d = np.linalg.norm(features - centers[c-1], axis=1)**2
        d2 = np.minimum(d2, d); probs = d2 / (d2.sum() + 1e-9)
        centers[c] = features[rng.choice(N, p=probs)]
    for _ in range(iters):
        dist = np.linalg.norm(features[:,None,:] - centers[None,:,:], axis=2)
        labels = dist.argmin(axis=1)
        new_centers = centers.copy()
        for i in range(k):
            sel = features[labels==i]
            if len(sel) > 0: new_centers[i] = sel.mean(axis=0)
        if np.allclose(new_centers, centers): break
        centers = new_centers
    dist = np.linalg.norm(features[:,None,:] - centers[None,:,:], axis=2)
    labels = dist.argmin(axis=1)
    return labels, centers

# ---------------- Poses and rendering ----------------
def generate_circular_control_sequence(center_t, radius=2.5, n_frames=180):
    poses = []
    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames
        offset = np.array([0.0, radius*np.sin(theta), radius*np.cos(theta)], np.float64)
        cam_t_i = center_t + offset
        target = np.array([0.0, 0.0, 3.5], np.float64)
        fwd = target - cam_t_i; fwd /= (np.linalg.norm(fwd) + 1e-9)
        yaw = np.degrees(np.arctan2(fwd[0], fwd[2])); pitch = np.degrees(np.arcsin(-fwd[1])); roll = 0.0
        cam_R_i = R_from_euler_xyz(pitch, yaw, roll)
        poses.append((cam_t_i, cam_R_i))
    return poses

def ring_world_pose(i, step_m, var_deg):
    base_z = 3.0
    z_i = base_z + i*step_m
    t_w = np.array([0.0, 0.0, z_i], np.float64)
    rxi = var_deg * math.sin(0.7*i); ryi = var_deg * math.cos(0.9*i); rzi = var_deg * math.sin(1.3*i)
    R_w = R_from_euler_xyz(rxi, ryi, rzi)
    return R_w, t_w

def obj_to_cam(R_w, t_w, R_cam, t_cam):
    R_oc = R_cam.T @ R_w; t_oc = R_cam.T @ (t_w - t_cam)
    return rvec_from_R(R_oc), t_oc.reshape(3,1)

def render_main_frame(ct, Rc, hfov, n_rings, step_m, var_deg, ring_poly, occ_on, K=None):
    if K is None:
        K, fx, fy, cx, cy = K_from_hfov(hfov)
    else:
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    main = np.full((BASE_H, BASE_W, 3), 28, np.uint8)
    cube_segs = cube_grid_segments(size=8.0, step=1.0)
    draw_cube_world(main, K, Rc, ct, cube_segs, color=(230,230,230))
    for i in range(n_rings):
        Rw, tw = ring_world_pose(i, step_m, var_deg)
        rvec, tvec = obj_to_cam(Rw, tw, Rc, ct)
        pts2d, _ = cv2.projectPoints(ring_poly, rvec, tvec, K, None)
        pts = np.round(pts2d).astype(np.int32).reshape(-1,1,2)
        col = (0, int(50 + 205*(i/max(1,n_rings-1))), 255)
        cv2.polylines(main, [pts], True, col, thickness=10, lineType=cv2.LINE_AA)
    if occ_on:
        x = int(0.45*BASE_W); y = int(0.60*BASE_H); w = int(0.18*BASE_W); h = int(0.10*BASE_H)
        cv2.rectangle(main, (x,y), (x+w, y+h), (200,50,200), thickness=-1)
    return K, fx, fy, cx, cy, main

# ---------------- Sequence runner ----------------
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

def run_control_sequence(cam_t, hfov, n_rings, step_m, var_deg, noise_std, blur_k, occ_on):
    run_dir = next_run_dir("algorithmOutputVisualizations")
    graph2d_dir = os.path.join(run_dir, "pca2d_frames"); os.makedirs(graph2d_dir, exist_ok=True)
    graph3d_dir = os.path.join(run_dir, "pca3d_frames"); os.makedirs(graph3d_dir, exist_ok=True)

    # Generate poses with larger baseline
    poses = generate_circular_control_sequence(cam_t.copy(), radius=2.5, n_frames=180)

    # Capture frames + sampled points
    ring_poly = ring_points3d(n=128)
    frames_bgr = []; frames_pts = []; stage1_frames = []
    K, fx, fy, cx, cy = K_from_hfov(hfov)
    for t_idx, (ct, Rc) in enumerate(poses):
        K, fx, fy, cx, cy, main = render_main_frame(ct, Rc, hfov, n_rings, step_m, var_deg, ring_poly, occ_on, K=K)
        main_noisy = add_noise_blur(main, noise_std=noise_std, blur_k=blur_k if blur_k%2==1 else max(1, blur_k-1))
        mask, band, pts, cids = detect_mask_and_sample_points(main_noisy, pts_per_contour=120)
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for p in pts.astype(np.int32):
            cv2.circle(vis, tuple(p), 2, (0,255,0), -1)
        cv2.putText(vis, f"Frame {t_idx:03d}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        frames_bgr.append(main_noisy); frames_pts.append(pts.copy()); stage1_frames.append(vis)

    T = len(frames_pts)
    if T == 0 or len(frames_pts[0]) == 0:
        return

    # Initialize tracks from first frame points
    N0 = len(frames_pts[0])
    # LK tracks with forward-backward check
    tracks = compute_tracks_lk(frames_bgr, frames_pts[0], fb_thresh=1.5)

    # Build magnitude history for temporal variance
    mag_hist = [np.array([], np.float32) for _ in range(N0)]
    for t_idx in range(1, T):
        for i in range(N0):
            p0 = tracks[i][t_idx-1]; p1 = tracks[i][t_idx]
            if not (np.isnan(p0).any() or np.isnan(p1).any()):
                mag = float(np.linalg.norm(p1 - p0))
                mag_hist[i] = np.append(mag_hist[i], mag)

    # Sequence-level unified features for labeling and visualization
    F_seq = track_features_sequence(tracks, BASE_W, BASE_H)   # normalized 6D features

    # Optional two-view triangulation for depth clustering
    labels = np.full((N0,), -1, np.int32)
    if USE_TRIANGULATION:
        try:
            if TRI_BASELINE_FRAMES is None:
                t0, t1 = 0, T//2
            else:
                t0, t1 = TRI_BASELINE_FRAMES
            ct0, Rc0 = poses[t0]
            ct1, Rc1 = poses[t1]

            Ktri, _, _, _, _ = K_from_hfov(hfov)
            P0 = Ktri @ np.hstack([Rc0.T, -Rc0.T @ ct0.reshape(3,1)])
            P1 = Ktri @ np.hstack([Rc1.T, -Rc1.T @ ct1.reshape(3,1)])

            p0 = np.array([tracks[i][t0] for i in range(N0)], np.float32)  # (N, 2)
            p1 = np.array([tracks[i][t1] for i in range(N0)], np.float32)
            valid = ~(np.isnan(p0).any(axis=1) | np.isnan(p1).any(axis=1))
            if valid.any():
                p0_v = p0[valid].T
                p1_v = p1[valid].T
                X4 = cv2.triangulatePoints(P0, P1, p0_v, p1_v)  # (4, M)
                X3 = (X4[:3] / (X4[3] + 1e-9)).T                # (M, 3)
                depth = X3[:, 2:3].astype(np.float32)           # (M, 1)
                d_labels, d_centers = kmeans_nd(depth, k=n_rings, iters=50, seed=42)
                known_idx = np.where(valid)[0]
                labels[known_idx] = d_labels

                # Propagate to unknown via nearest-centroid in F_seq space
                centroids = []
                for c in range(n_rings):
                    sel = (labels == c)
                    if sel.any():
                        centroids.append(F_seq[sel].mean(axis=0))
                    else:
                        centroids.append(np.zeros((F_seq.shape[1],), np.float32))
                centroids = np.vstack(centroids).astype(np.float32)

                unknown = np.where(labels < 0)[0]
                if len(unknown) > 0:
                    D = np.linalg.norm(F_seq[unknown][:,None,:] - centroids[None,:,:], axis=2)
                    labels[unknown] = D.argmin(axis=1).astype(np.int32)
            else:
                # Fallback: k-means on F_seq if no valid triangulation set
                labels, _ = kmeans_nd(F_seq, k=n_rings, iters=50, seed=42)
        except Exception:
            labels, _ = kmeans_nd(F_seq, k=n_rings, iters=50, seed=42)
    else:
        labels, _ = kmeans_nd(F_seq, k=n_rings, iters=50, seed=42)

    # Stable PCA basis from cumulative per-frame features across entire sequence
    G_all = []
    for t in range(T):
        G_t = np.zeros((N0, 6), np.float32)
        for i in range(N0):
            tr = tracks[i][:t+1]
            valid_i = ~np.isnan(tr).any(axis=1)
            pts = tr[valid_i]
            if len(pts) < 2:
                continue
            d = np.diff(pts, axis=0)
            mags = np.linalg.norm(d, axis=1) + 1e-6
            dirs = d / mags[:, None]
            G_t[i, 0] = mags.mean()
            G_t[i, 1:3] = dirs.mean(axis=0)
            G_t[i, 3:5] = pts.mean(axis=0) / np.array([BASE_W, BASE_H], np.float32)
            G_t[i, 5] = mags.std()
        G_all.append(G_t)

    X = np.vstack(G_all)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Z3_all, mu_all, Wpca3 = pca_project(X, n_components=3)  # stable 3D basis
    Z2_all = Z3_all[:, :2]
    # Save per-frame 2D PCA scatter frames using stable basis with consistent labels
    pca2d_frames = []
    for t_idx in range(T):
        Z2_t = Z2_all[t_idx*N0:(t_idx+1)*N0]
        plt.figure(figsize=(8,4.5), dpi=160)
        for c in range(n_rings):
            sel = (labels == c)
            if sel.any():
                plt.scatter(Z2_t[sel,0], Z2_t[sel,1], s=8, label=f"obj {c+1}")
        plt.title(f"PCA(2D) stable basis (frame {t_idx})")
        plt.xlabel("PC1"); plt.ylabel("PC2"); plt.grid(True, alpha=0.3); plt.legend(markerscale=2, fontsize=7)
        png_path = os.path.join(graph2d_dir, f"pca2d_{t_idx:03d}.png")
        plt.savefig(png_path, bbox_inches='tight'); plt.close()
        img = cv2.imread(png_path)
        if img is None: img = np.zeros((540,960,3), np.uint8)
        pca2d_frames.append(img)

    # Optional 3D PCA snapshot using stable basis on the last frame slice
    if _HAS_PLOTLY:
        try:
            Z3_last = Z3_all[(T-1)*N0 : T*N0]
            label_last = labels
            fig = px.scatter_3d(x=Z3_last[:,0], y=Z3_last[:,1], z=Z3_last[:,2], color=label_last.astype(str),
                                title="Stable PCA(3D) of cumulative features (last frame slice)")
            fig.write_html(os.path.join(run_dir, "pca3d_last_frame.html"))
        except Exception:
            pass
    else:
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            Z3_last = Z3_all[(T-1)*N0 : T*N0]
            label_last = labels
            frames3d = []
            for az in range(0, 360, 6):
                fig = plt.figure(figsize=(6,6), dpi=160)
                ax = fig.add_subplot(111, projection='3d')
                for c in range(n_rings):
                    sel = (label_last == c)
                    if sel.any():
                        ax.scatter(Z3_last[sel,0], Z3_last[sel,1], Z3_last[sel,2], s=8, label=f"obj {c+1}")
                ax.set_title("Stable PCA(3D) last frame"); ax.view_init(elev=20, azim=az)
                ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
                png = os.path.join(graph3d_dir, f"pca3d_{az:03d}.png")
                plt.savefig(png, bbox_inches='tight'); plt.close()
                img = cv2.imread(png)
                if img is not None: frames3d.append(img)
            write_video(os.path.join(run_dir, "pca3d_last_frame.mp4"), frames3d, fps=20)
        except Exception:
            pass

    # Colored points video using final labels
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

    # Fit ellipse per cluster (with optional spatial post-split on final frame)
    summary = frames_bgr[-1].copy()
    final_ellipses = []
    # Positions at final frame
    final_pts = np.array([tracks[i][-1] for i in range(N0)])
    valid_final = ~np.isnan(final_pts).any(axis=1)

    for c in range(n_rings):
        tids = np.where(labels==c)[0]
        tids = tids[valid_final[tids]]
        if len(tids) == 0:
            continue

        # Optional spatial post-split
        components = [tids]
        if _HAS_SKLEARN and len(tids) >= DBSCAN_MIN_SAMPLES:
            pts = final_pts[tids]
            db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(pts)
            spatial_labels = db.labels_
            uniq = sorted(set(int(v) for v in spatial_labels.tolist()) - {-1})
            comps = []
            for sp in uniq:
                comp_idx = tids[spatial_labels == sp]
                if len(comp_idx) > 0:
                    comps.append(comp_idx)
            if len(comps) > 0:
                components = comps

        for comp in components:
            # Accumulate all points across time for these track IDs
            P_list = []
            for tid in comp:
                tr = tracks[tid]
                P_list.append(tr[~np.isnan(tr).any(axis=1)])
            if len(P_list) == 0: 
                continue
            P = np.vstack(P_list).astype(np.float32)
            if len(P) < 5:
                continue
            cnt = P.reshape(-1,1,2).astype(np.int32)
            try: e = cv2.fitEllipseAMS(cnt)
            except:
                try: e = cv2.fitEllipse(cnt)
                except: 
                    continue
            final_ellipses.append(e)
            cv2.ellipse(summary, e, (0,255,0), 2, cv2.LINE_AA)

    # Save artifacts
    write_video(os.path.join(run_dir, "stage1_sampled_points.mp4"), stage1_frames, fps=30)
    write_video(os.path.join(run_dir, "stage2_color_clusters.mp4"), stage2_frames, fps=30)
    write_video(os.path.join(run_dir, "pca2d_frames.mp4"), pca2d_frames, fps=30)
    cv2.imwrite(os.path.join(run_dir, "final_object_assignments.png"), summary)

# ---------------- Main UI ----------------
def main():
    cv2.namedWindow('Sim', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Sim', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cam_t = np.array([4.93, -3.20, -2.03], np.float64)
    cam_rx, cam_ry, cam_rz = -16.3, -45.6, -2.1
    hfov = 78.0

    n_rings = 4; step_m = 3.0; var_deg = 12.0
    noise_std = 7; blur_k = 3; occ_on = 0; help_on = True

    TX_RATE = 1.5; TY_RATE = 1.5; TZ_RATE = 1.6; ANG_RATE = 120.0
    ring_poly = ring_points3d(n=128)
    cube_segs = cube_grid_segments(size=8.0, step=1.0)

    tprev = time.perf_counter(); fps = 0.0
    _fallback_canvas = np.zeros((BASE_H, BASE_W, 3), np.uint8)
    seq_running = False

    while True:
        tnow = time.perf_counter(); dt = max(1e-3, tnow - tprev); tprev = tnow
        fps = (1.0/dt) if fps == 0 else (1-0.1)*fps + 0.1*(1.0/dt)

        k = cv2.waitKey(1) & 0xFFFF
        if k == 27: break
        if k == ord('t') and not seq_running:
            seq_running = True
            try:
                run_control_sequence(cam_t.copy(), hfov, n_rings, step_m, var_deg, noise_std, blur_k, occ_on)
            except Exception as e:
                print("Sequence error:", e)
            seq_running = False

        # Translate
        R_cam = R_from_euler_xyz(cam_rx, cam_ry, cam_rz)
        d = np.zeros(3, np.float64)
        if k == ord('a') or k in {81,65361}:   d[0] -= TX_RATE*dt
        if k == ord('d') or k in {83,65363}:   d[0] += TX_RATE*dt
        if k == ord('w') or k in {82,65362}:   d[1] -= TY_RATE*dt
        if k == ord('s') or k in {84,65364}:   d[1] += TY_RATE*dt
        if k == ord('q'):                      d[2] += TZ_RATE*dt
        if k == ord('e'):                      d[2] -= TZ_RATE*dt
        cam_t += R_cam @ d

        # Rotate
        if k == ord('j'): cam_ry -= ANG_RATE*dt
        if k == ord('l'): cam_ry += ANG_RATE*dt
        if k == ord('u'): cam_rz -= ANG_RATE*dt
        if k == ord('o'): cam_rz += ANG_RATE*dt
        if k == ord('k'): cam_rx -= ANG_RATE*dt
        if k == ord('i'): cam_rx += ANG_RATE*dt

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

        # Tile sizing with HighGUI guard
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
            Rw, tw = ring_world_pose(i, step_m, var_deg)
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

        # Quick stages + data
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

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
