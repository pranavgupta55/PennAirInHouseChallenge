#!/usr/bin/env python3
"""
Ellipse propagation + depth-aware ellipse refinement demo (self-contained)

This script:
 - renders synthetic rings and produces per-frame rim-band detections
 - samples contour points per frame
 - seeds LK tracks from frame-0 points and finds stable seed tracks
 - fits initial image-space ellipses to those seed tracks and extends them
   across all frames by searching along the projected ellipse perimeter
 - **GROUPS 2D trajectories based on their initial fitted ellipse parameters.**
 - triangulates 3D positions for each ellipse's tracked points using camera
   poses (parallax) and fits a 3D circle in the best-fit plane
 - reprojects the 3D circle to each image to get a refined, depth-aware ellipse
 - clusters final ellipse descriptors [cx,cy,w,h] and outputs visuals

Requirements: Python3, numpy, OpenCV (cv2), matplotlib. scikit-learn optional.

Drop this file into a fresh repo and run. Output goes to algorithmOutputVisualizations/ and debugInfo/.
"""
import os, glob, math, cv2, time, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# -------------------- USER-CONTROLLED WORLD / ALGORITHM SETTINGS --------------------
BASE_W, BASE_H = 960, 540
HFOV = 78.0
CIRC_RADIUS = 0.5
N_FRAMES = 180

N_RINGS = 4
STEP_M = 0.9
VAR_DEG = 12.0
WORLD_SIZE = 8.0
LATERAL_FRACTIONS = (0.75, 0.25, 0.75, 0.25)  # requested alternating positions
HEIGHT_MODE = 'jitter'
VERTICAL_BAND = (2.5, 5.0)
VERTICAL_JITTER = 0.25
RING_DIAM_M = 0.9144
RING_RAD_M = RING_DIAM_M / 2.0

NOISE_STD = 7
BLUR_K = 3
RANDOM_SEED = 42

PTS_PER_CONTOUR = 60
MIN_CONTOUR_LEN = 20

LK_FB_THRESH = 1.5
LK_WIN = (21, 21)
LK_MAX_LEVEL = 3

ELLIPSE_INIT_MIN_FRAMES = 5
ELLIPSE_MATCH_DIST_PX = 12.0
ELLIPSE_SAMPLE_RES = 360

MIN_TRAJECTORY_LEN = 8

# DBSCAN settings for final descriptors
DBSCAN_EPS = 40.0
DBSCAN_MIN_SAMPLES = 2
# DBSCAN settings for trajectory grouping (based on initial 2D ellipse centers)
GROUP_DBSCAN_EPS_PX = 100.0
GROUP_DBSCAN_MIN_SAMPLES = 5 # Needs enough points to form a solid ring track

RUN_DIR_BASE = "algorithmOutputVisualizations"
DEBUG_DIR_BASE = "debugInfo"

# -------------------- Utilities --------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def next_run_dir(base=RUN_DIR_BASE):
    ensure_dir(base)
    existing = sorted(glob.glob(os.path.join(base, "run_*")))
    idx = 1
    if existing:
        try:
            idx = int(os.path.basename(existing[-1]).split("_")[-1]) + 1
        except Exception:
            idx = len(existing) + 1
    run_dir = os.path.join(base, f"run_{idx:03d}")
    ensure_dir(run_dir)
    return run_dir

def next_debug_dir(base=DEBUG_DIR_BASE):
    ensure_dir(base)
    existing = sorted(glob.glob(os.path.join(base, "run_*")))
    idx = 1
    if existing:
        try:
            idx = int(os.path.basename(existing[-1]).split("_")[-1]) + 1
        except Exception:
            idx = len(existing) + 1
    d = os.path.join(base, f"run_{idx:03d}")
    ensure_dir(d)
    return d

def write_video(path, frames, fps=30):
    if not frames:
        return
    h,w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(path, fourcc, fps, (w,h))
    for fr in frames:
        img = fr
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w,h))
        vw.write(img)
    vw.release()

def filter_abnormal_descriptors(descs, traj_indices, W, H):
    """
    Removes descriptors [cx, cy, w, h] that have unreasonable values for center or size.
    An 'unreasonable' value suggests a poor ellipse fit artifact.
    """
    if descs.shape[0] == 0:
        return np.zeros((0,4), dtype=np.float32), []

    cx, cy, w, h = descs[:,0], descs[:,1], descs[:,2], descs[:,3]
    
    # 1. Center must be within a reasonable boundary (e.g., 0.5 image size margin)
    W_margin = 0.5 * W
    H_margin = 0.5 * H
    cx_ok = (cx > -W_margin) & (cx < W + W_margin)
    cy_ok = (cy > -H_margin) & (cy < H + H_margin)
    
    # 2. Width/Height (major/minor axis) must not be ridiculously large or non-positive
    max_diag = np.sqrt(W*W + H*H)
    # Allow size up to 2x image diagonal (very conservative upper bound)
    max_size_allowed = 2.0 * max_diag 
    min_size_allowed = 5.0 # Minimum reasonable axis length in pixels
    
    w_ok = (w > min_size_allowed) & (w < max_size_allowed)
    h_ok = (h > min_size_allowed) & (h < max_size_allowed)

    # All criteria must be met
    valid_mask = cx_ok & cy_ok & w_ok & h_ok
    
    filtered_descs = descs[valid_mask]
    filtered_traj_indices = [traj_indices[i] for i, valid in enumerate(valid_mask) if valid]
    
    print(f"Filtered {descs.shape[0] - filtered_descs.shape[0]} abnormal descriptors from {descs.shape[0]} total.")
    
    return filtered_descs, filtered_traj_indices

# -------------------- Geometry / camera --------------------
def K_from_hfov(hfov, w=BASE_W, h=BASE_H):
    hfov = math.radians(hfov)
    fx = (w/2) / math.tan(hfov/2)
    fy = fx
    cx, cy = w/2, h/2
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float64)

def R_from_euler_xyz(rx,ry,rz):
    rx,ry,rz = map(math.radians,[rx,ry,rz])
    Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
    Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
    Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx

def ring_points3d(radius_m=RING_RAD_M, n=128):
    t = np.linspace(0, 2*np.pi, n, endpoint=True)
    return np.vstack([radius_m*np.cos(t), radius_m*np.sin(t), np.zeros_like(t)]).T.astype(np.float64)

def obj_to_cam(R_w, t_w, R_cam, t_cam):
    R_oc = R_cam.T @ R_w
    t_oc = R_cam.T @ (t_w - t_cam)
    rvec, _ = cv2.Rodrigues(R_oc)
    return rvec.astype(np.float64), t_oc.reshape(3,1)

def ring_world_pose(i, step_m, var_deg,
                    lateral_fractions=LATERAL_FRACTIONS,
                    world_size=WORLD_SIZE,
                    height_mode=HEIGHT_MODE,
                    vertical_band=VERTICAL_BAND,
                    vertical_jitter=VERTICAL_JITTER,
                    seed=None):
    rng = np.random.default_rng(int(seed)+i) if seed is not None else np.random.default_rng()
    frac = lateral_fractions[i % len(lateral_fractions)]
    x = -world_size/2.0 + frac * world_size
    base_z = 3.0
    base_z_i = base_z + i * step_m
    min_z, max_z = vertical_band
    if height_mode == 'uniform':
        z_i = float(rng.uniform(min_z, max_z))
    else:
        jitter = float(rng.uniform(-vertical_jitter, vertical_jitter))
        z_i = base_z_i + jitter
        z_i = float(np.clip(z_i, min_z, max_z))
    y = 0.0
    t_w = np.array([x,y,z_i], np.float64)
    rxi = var_deg * math.sin(0.7 * i)
    ryi = var_deg * math.cos(0.9 * i)
    rzi = var_deg * math.sin(1.3 * i)
    R_w = R_from_euler_xyz(rxi, ryi, rzi)
    return R_w, t_w

def compute_camera_projection(K, R_cam, cam_t):
    """
    Compute 3x4 projection matrix P = K [R^T | -R^T * t]
    where R_cam is rotation of camera in world coordinates and cam_t its world position.
    """
    P = K @ np.hstack([R_cam.T, -R_cam.T @ cam_t.reshape(3,1)])
    return P

# -------------------- Rendering & detection --------------------
def add_noise_blur(img, noise_std=0, blur_k=0):
    out = img.astype(np.float32)
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, out.shape).astype(np.float32)
        out = np.clip(out + noise, 0, 255)
    out = out.astype(np.uint8)
    if blur_k > 0 and blur_k % 2 == 1:
        out = cv2.GaussianBlur(out, (blur_k, blur_k), 0)
    return out

def detect_rim_band(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:,:,1]
    _, mask = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    band = cv2.morphologyEx(mask2, cv2.MORPH_GRADIENT, k)
    contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours, S, mask2, band

def sample_contour_points(contour, n_pts=PTS_PER_CONTOUR):
    contour = contour.reshape(-1,2).astype(np.float32)
    if len(contour) < 3:
        return np.zeros((0,2), np.float32)
    diffs = np.diff(contour, axis=0, prepend=contour[-1:])
    lens = np.linalg.norm(diffs, axis=1)
    arc = np.cumsum(lens); total = arc[-1] if arc.size>0 else 0.0
    if total < 1e-6:
        return contour[:n_pts] if len(contour)>=n_pts else contour
    sample_arcs = np.linspace(0, total, n_pts, endpoint=False)
    idxs = np.searchsorted(arc, sample_arcs); idxs = np.clip(idxs, 0, len(contour)-1)
    return contour[idxs]

# -------------------- LK seed tracking --------------------
def compute_tracks_lk(frames_bgr, init_points, fb_thresh=LK_FB_THRESH):
    T = len(frames_bgr)
    N0 = len(init_points)
    tracks = [np.full((T,2), np.nan, np.float32) for _ in range(N0)]
    if T == 0 or N0 == 0: return tracks
    for i in range(N0): tracks[i][0] = init_points[i]
    I_prev = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    p_prev = init_points.reshape(-1,1,2).astype(np.float32)
    lk_params = dict(winSize=LK_WIN, maxLevel=LK_MAX_LEVEL,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3))
    for t in range(1, T):
        I_curr = cv2.cvtColor(frames_bgr[t], cv2.COLOR_BGR2GRAY)
        p_next, st, err = cv2.calcOpticalFlowPyrLK(I_prev, I_curr, p_prev, None, **lk_params)
        p_back, st2, _ = cv2.calcOpticalFlowPyrLK(I_curr, I_prev, p_next, None, **lk_params)
        fb_err = np.linalg.norm(p_prev - p_back, axis=2).reshape(-1)
        ok = (st.reshape(-1) > 0) & (st2.reshape(-1) > 0) & (fb_err < fb_thresh)
        for i in range(N0):
            if ok[i] and not np.any(np.isnan(p_next[i,0])):
                tracks[i][t] = p_next[i,0]
            else:
                tracks[i][t] = np.nan
        I_prev = I_curr; p_prev = p_next
    return tracks

# -------------------- ellipse helpers --------------------
def fit_ellipse_to_points(pts):
    if pts is None or len(pts) < 5:
        return None
    try:
        cnt = pts.reshape(-1,1,2).astype(np.int32)
        e = cv2.fitEllipse(cnt)
        return e
    except Exception:
        return None

def ellipse_samples_from_params(e, n=ELLIPSE_SAMPLE_RES):
    (cx,cy),(MA,ma),ang = e
    a = MA / 2.0; b = ma / 2.0
    theta = math.radians(ang)
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.vstack([a * np.cos(t), b * np.sin(t)]).T
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    pts_rot = pts @ R.T
    pts_rot[:,0] += cx; pts_rot[:,1] += cy
    return pts_rot

def distance_point_to_samples(pt, samples):
    d = np.linalg.norm(samples - pt.reshape(1,2), axis=1)
    return d.min()

# -------------------- triangulation and 3D circle fit --------------------
def triangulate_pair(P0, P1, p0, p1):
    """
    p0, p1: arrays shape (2,) or (2,N)
    P0, P1: 3x4 projection matrices
    returns: (3,N) points in world coordinates
    """
    # cv2.triangulatePoints expects float32/64 and 2xN arrays
    p0a = np.array(p0, dtype=np.float64).reshape(2,-1)
    p1a = np.array(p1, dtype=np.float64).reshape(2,-1)
    X4 = cv2.triangulatePoints(P0, P1, p0a, p1a)  # (4,N)
    X3 = (X4[:3] / (X4[3] + 1e-12))
    return X3  # shape (3,N)

def triangulate_trajectory_points(trajectory, poses, K, min_pairs=1):
    """
    trajectory: list of (t,x,y)
    poses: list of (cam_t, R_cam)
    K: intrinsics 3x3
    Returns: Nx3 triangulated points in world coordinates (may have NaNs if failed)
    """
    if len(trajectory) < 2:
        return np.zeros((0,3), np.float32)
    T = len(poses)
    
    MAX_DISTANCE_FROM_ORIGIN = 20.0 # Strict limit for ill-conditioned points

    # Build map t->(x,y)
    t_to_xy = {int(t): np.array([x,y], dtype=np.float64) for (t,x,y) in trajectory}
    times = sorted(t_to_xy.keys())
    pts_3d = []
    # Precompute projection matrices
    P_mats = []
    for (ct, Rc) in poses:
        P_mats.append(compute_camera_projection(K, Rc, ct))
    # choose pairs: for each time, pair with time separated by at least sep frames
    # Use a large separation for better baseline: T/4
    sep = max(1, len(poses) // 4) 
    for i, t0 in enumerate(times):
        for t1 in times[i+1:]:
            if abs(t1 - t0) < sep:
                continue
            p0 = t_to_xy[t0]; p1 = t_to_xy[t1]
            P0 = P_mats[t0]; P1 = P_mats[t1]
            try:
                X3 = triangulate_pair(P0, P1, p0, p1)  # shape (3,1)
                x3 = X3[:,0]
                
                # Sanity Check 1: Reject points too far from the origin
                if np.linalg.norm(x3) > MAX_DISTANCE_FROM_ORIGIN:
                    continue
                
                # Sanity Check 2: check depth positive in both camera frames:
                ct0, Rc0 = poses[t0]
                cam0_coords = Rc0.T @ (x3.reshape(3,) - ct0)
                ct1, Rc1 = poses[t1]
                cam1_coords = Rc1.T @ (x3.reshape(3,) - ct1)
                if cam0_coords[2] <= 0 or cam1_coords[2] <= 0:
                    continue
                pts_3d.append(x3)
            except Exception:
                continue
    if len(pts_3d) < min_pairs:
        return np.zeros((0,3), np.float32)
    return np.vstack(pts_3d).astype(np.float32)

def fit_plane_pca(X):
    """
    Fit plane to 3D points X (N,3) via SVD. Return plane origin (mean), basis e1,e2 (2 orthonormal axes), normal.
    """
    if X.shape[0] < 3:
        return None, None, None, None
    mu = X.mean(axis=0)
    Xc = X - mu
    
    # Use float32 to match cv2/numpy default in other parts, 
    # but SVD is generally more robust on float64, keeping it float64 as per numpy default
    try:
        U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    except Exception:
        return None, None, None, None

    # principal directions: Vt[0], Vt[1]; normal = Vt[2]
    e1 = Vt[0,:]
    e2 = Vt[1,:]
    normal = Vt[2,:]
    return mu, e1, e2, normal

def project_points_to_plane_coords(X, plane_origin, e1, e2):
    """
    Express X in plane coordinates (u,v) such that X ≈ plane_origin + u*e1 + v*e2
    """
    Xc = X - plane_origin
    u = Xc @ e1
    v = Xc @ e2
    return np.vstack([u, v]).T  # (N,2)

def fit_circle_2d_kasa(pts2d):
    """
    Fit circle to 2D points using algebraic least-squares (Kasa).
    pts2d: (N,2)
    returns center (a,b), radius r or None if fail.
    """
    if pts2d.shape[0] < 3:
        return None, None
    x = pts2d[:,0]; y = pts2d[:,1]
    A = np.vstack([2*x, 2*y, np.ones_like(x)]).T  # (N,3)
    b = x**2 + y**2
    try:
        # lstsq is generally robust
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        a, b0, c = sol
        # Check for non-physical radius (imaginary)
        if (a*a + b0*b0 + c) < 0:
             return None, None
        r = math.sqrt(max(0.0, a*a + b0*b0 + c))
        center = np.array([a,b0], dtype=np.float32)
        return center, float(r)
    except Exception:
        return None, None

def build_3d_circle_from_triangulated(X3):
    """
    Given a set of 3D points from triangulation, fit plane and then 2D circle in that plane.
    Returns dict with center3D, radius, plane_origin, e1,e2, normal or None if failed.
    """
    if X3.shape[0] < 3:
        return None
    plane_origin, e1, e2, normal = fit_plane_pca(X3)
    if plane_origin is None:
        return None
    # normalize axes to unit
    e1n = e1 / (np.linalg.norm(e1) + 1e-12)
    e2n = e2 / (np.linalg.norm(e2) + 1e-12)
    pts2d = project_points_to_plane_coords(X3, plane_origin, e1n, e2n)
    
    # Filter out potential outliers in the projected 2D space before circle fit
    # This addresses the possibility that a few far-off 3D points survived triangulation sanity checks
    # and would skew the 2D circle fit.
    if pts2d.shape[0] > 0:
        mean_2d = pts2d.mean(axis=0)
        dist_2d = np.linalg.norm(pts2d - mean_2d, axis=1)
        # Assuming all points belong to a ring of approx. RING_RAD_M (0.4572m)
        # Use 2x expected diameter as a generous outlier threshold
        outlier_threshold = 2.0 * RING_DIAM_M
        inlier_mask = dist_2d < outlier_threshold
        pts2d_filtered = pts2d[inlier_mask]
        
        if pts2d_filtered.shape[0] < 3:
            return None
        pts2d = pts2d_filtered
        
    center2d, radius2d = fit_circle_2d_kasa(pts2d)
    if center2d is None:
        return None
    
    # Sanity check the fitted radius against the known ring radius (RING_RAD_M)
    # The fitted circle should be close to the actual radius.
    # Set a generous tolerance, e.g., 5x the actual radius for robustness against projection effects
    if radius2d > 5.0 * RING_RAD_M or radius2d < 0.1 * RING_RAD_M:
         return None
         
    center3d = plane_origin + center2d[0]*e1n + center2d[1]*e2n
    return {
        'center3d': center3d,
        'radius': radius2d,
        'plane_origin': plane_origin,
        'e1': e1n,
        'e2': e2n,
        'normal': normal
    }

def sample_3d_circle(circle3d, n=256):
    """
    Return Nx3 points of 3D circle parameterized on its plane
    """
    center = circle3d['center3d']
    r = circle3d['radius']
    e1 = circle3d['e1']; e2 = circle3d['e2']
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.array([center + r*(math.cos(tt)*e1 + math.sin(tt)*e2) for tt in t], dtype=np.float32)
    return pts

# -------------------- main pipeline --------------------
def color_list(n):
    rng = np.random.default_rng(12345)
    cols = (255 * rng.random((n,3))).astype(np.uint8).tolist()
    return cols

def draw_trajectories_overlay(img, trajectories, colors, draw_points=True, draw_ellipses=True):
    out = img.copy()
    for i,tr in enumerate(trajectories):
        col = tuple(int(x) for x in colors[i % len(colors)])
        if draw_points:
            # Trajectories are now lists of (t, x, y) tuples from all points
            points_at_t = [(x,y) for (t,x,y) in tr['trajectory'] if t == len(out)] # t is not index, but time index
            
            # Need to iterate over the time-mapped trajectory points
            t_idx = int(tr.get('ref_frame_idx', -1)) # use ref_frame_idx for the single frame overlay if available

            for (t,x,y) in tr['trajectory']:
                if t == t_idx:
                    px,py = int(round(x)), int(round(y))
                    if 0 <= px < out.shape[1] and 0 <= py < out.shape[0]:
                         cv2.circle(out, (px,py), 2, col, -1)
                
            # For the single frame overlay, simplify to just drawing points for all frames
            # This logic is too complex for a single frame, let's fix the call to draw_trajectories_overlay
            # The previous code only drew points for the frame where the trajectory *ended*.
            # For debugging single frames, we should draw all points visible in that frame.
            
            # Let's fix this for the single frame overlay:
            # We want to show all points for the current trajectory *at a given frame t_idx*
            if 'trajectory_map' in tr:
                if t_idx in tr['trajectory_map']:
                    for (x,y) in tr['trajectory_map'][t_idx]:
                        px,py = int(round(x)), int(round(y))
                        if 0 <= px < out.shape[1] and 0 <= py < out.shape[0]:
                             cv2.circle(out, (px,py), 2, col, -1)
                    
        if draw_ellipses and 'refined_ellipse' in tr and tr['refined_ellipse'] is not None:
            e = tr['refined_ellipse']
            # e is cv2 ellipse tuple
            cv2.ellipse(out, ( (int(round(e[0][0])), int(round(e[0][1]))),
                               (int(round(e[1][0])), int(round(e[1][1]))),
                               float(e[2]) ), color=col, thickness=2, lineType=cv2.LINE_AA)
    return out

def group_trajectories_by_initial_fit(trajectories, W, H):
    """
    Groups individual point trajectories into Super-Trajectories representing
    each physical ring, based on the proximity of their initial fitted 2D ellipses.
    """
    if not trajectories:
        return []
        
    # 1. Collect descriptors and indices
    initial_descs = []
    valid_traj_indices = []
    for i, tr in enumerate(trajectories):
        e = tr.get('fitted_ellipse')
        if e is not None:
            initial_descs.append(ellipse_descriptor_from_params(e))
            valid_traj_indices.append(i)
            
    if not initial_descs:
        return []
        
    descs = np.vstack(initial_descs)
    
    # Normalize centers by image size (W,H) for better clustering scale
    # Normalize descriptors: [cx/W, cy/H, w/W, h/H] - using just centers might be enough
    centers = descs[:, 0:2]
    centers_norm = centers.copy()
    centers_norm[:,0] /= W
    centers_norm[:,1] /= H

    # 2. Cluster the ellipse centers to group trajectories for the same ring
    # Use normalized centers for distance calculation, but use DBSCAN_EPS as a scale factor
    eps_norm = GROUP_DBSCAN_EPS_PX / float((W + H) / 2.0)
    
    if SKLEARN_OK:
        # Use DBSCAN on normalized centers
        db = DBSCAN(eps=eps_norm, min_samples=GROUP_DBSCAN_MIN_SAMPLES, metric='euclidean')
        labels = db.fit_predict(centers_norm)
    else:
        # Fallback to KMeans (simple clustering for fixed N_RINGS)
        def kmeans_simple(X, k):
             # (implementation from main, simplified)
            rng = np.random.default_rng(42)
            N = X.shape[0]
            k = min(k, N)
            if k == 0: return np.zeros(N, dtype=np.int64) - 1 # No clusters
            centers = X[rng.choice(N, size=k, replace=False)]
            for _ in range(50):
                d = np.linalg.norm(X[:,None,:] - centers[None,:,:], axis=2)
                labs = d.argmin(axis=1)
                newc = np.array([X[labs==i].mean(axis=0) if np.any(labs==i) else centers[i] for i in range(k)])
                if np.allclose(newc, centers): break
                centers = newc
            return d.argmin(axis=1)
            
        k = max(1, N_RINGS)
        labels = kmeans_simple(centers_norm, k)


    # 3. Aggregate individual trajectories into Super-Trajectories
    unique_labels = np.unique(labels)
    super_trajectories = []
    print(f"Clustered {len(valid_traj_indices)} tracks into {len(unique_labels)} groups.")

    for label in unique_labels:
        if label == -1: continue # Ignore noise
        
        cluster_indices = [valid_traj_indices[i] for i, l in enumerate(labels) if l == label]
        
        # Merge trajectory points (t, x, y) from all tracks in the cluster
        merged_trajectory = []
        for traj_idx in cluster_indices:
            merged_trajectory.extend(trajectories[traj_idx]['trajectory'])

        # Create a dictionary to map time index to a list of (x,y) points
        traj_map = {}
        for (t,x,y) in merged_trajectory:
            if t not in traj_map:
                traj_map[t] = []
            traj_map[t].append((x,y))

        # Build the final Super-Trajectory object
        # The 'trajectory' now holds ALL points from the cluster for 3D estimation
        super_traj = {
            'label': int(label),
            'cluster_indices': cluster_indices,
            'trajectory_map': traj_map, # For easier visualization access
            'trajectory': merged_trajectory # List of (t, x, y) tuples
        }
        
        # If possible, include a reference frame index for single-frame visualization
        if traj_map:
             super_traj['ref_frame_idx'] = sorted(traj_map.keys())[len(traj_map)//2]
        
        super_trajectories.append(super_traj)
        
    return super_trajectories

def refine_ellipses_with_depth(trajectories, poses, K):
    """
    For each Super-Trajectory:
      - triangulate 3D points from tracked 2D observations (now many points per frame)
      - fit 3D circle in best-fit plane
      - sample 3D circle and project into a reference image
      - fit an ellipse to that projection -> refined ellipse (depth-aware)
    Returns updated trajectories with 'refined_ellipse' and 'circle3d' fields (if successful)
    """
    T = len(poses)
    refined_count = 0
    
    for tr in trajectories:
        traj = tr['trajectory']  # list of (t,x,y) from all clustered tracks
        
        # Check if enough distinct observations exist (min frames and min total points)
        if len(set([t for (t,_,_) in traj])) < 2 or len(traj) < 10:
            tr['refined_ellipse'] = None
            tr['circle3d'] = None
            continue
            
        # 1. Triangulate many pairs
        # The original triangulate_trajectory_points function is designed for a single track point,
        # but since 'traj' contains all the points, it will naturally triangulate all pairs, 
        # which is what we want for a robust fit.
        X3 = triangulate_trajectory_points(traj, poses, K, min_pairs=10) # Require more pairs for a dense point cloud
        
        if X3.shape[0] < 10: # Require a dense enough 3D cloud
            tr['refined_ellipse'] = None
            tr['circle3d'] = None
            continue
            
        # 2. Fit 3D circle
        circle3d = build_3d_circle_from_triangulated(X3)
        if circle3d is None:
            tr['refined_ellipse'] = None
            tr['circle3d'] = None
            continue
            
        # 3. Sample 3D circle and reproject to a reference frame
        sample3d = sample_3d_circle(circle3d, n=256)  # (N,3)
        ref_t = tr.get('ref_frame_idx', sorted([t for (t,_,_) in traj])[len(traj)//2]) # Use the stored ref frame if available
        
        ct_ref, Rc_ref = poses[ref_t]
        P_ref = compute_camera_projection(K, Rc_ref, ct_ref)
        
        # Project sample3d into image
        X4 = np.vstack([sample3d.T, np.ones((1, sample3d.shape[0]))])
        proj = (P_ref @ X4)
        proj_xy = (proj[:2] / (proj[2:3] + 1e-9)).T  # (N,2)
        
        # 4. Fit ellipse to the projected samples
        e_proj = fit_ellipse_to_points(proj_xy.astype(np.float32))
        
        if e_proj is not None:
            tr['refined_ellipse'] = e_proj
            tr['circle3d'] = circle3d
            refined_count += 1
        else:
            tr['refined_ellipse'] = None
            tr['circle3d'] = None
            
    return refined_count

def ellipse_descriptor_from_params(e):
    (cx,cy),(MA,ma),ang = e
    return np.array([float(cx), float(cy), float(MA), float(ma)], dtype=np.float32)

def normalize_descriptors(descs, W=BASE_W, H=BASE_H):
    out = np.empty_like(descs, dtype=np.float32)
    out[:,0] = descs[:,0] / float(W)
    out[:,1] = descs[:,1] / float(H)
    out[:,2] = descs[:,2] / float(W)
    out[:,3] = descs[:,3] / float(H)
    return out

# -------------------- Main --------------------
def main():
    run_dir = next_run_dir(RUN_DIR_BASE)
    debug_dir = next_debug_dir(DEBUG_DIR_BASE)
    print("Run dir:", run_dir)
    print("Debug dir:", debug_dir)
    np.random.seed(RANDOM_SEED)

    # generate camera poses
    cam_center = np.array([4.93, -3.20, -2.03], np.float64)
    def generate_circular_control_sequence(center_t, radius=CIRC_RADIUS, n_frames=N_FRAMES):
        poses = []
        for i in range(n_frames):
            theta = 2 * math.pi * i / n_frames
            offset = np.array([0.0, radius*math.sin(theta), radius*math.cos(theta)], np.float64)
            cam_t_i = center_t + offset
            target = np.array([0.0, 0.0, 4.0], np.float64)
            fwd = target - cam_t_i; fwd /= (np.linalg.norm(fwd) + 1e-9)
            yaw = math.degrees(math.atan2(fwd[0], fwd[2]))
            pitch = math.degrees(math.asin(-fwd[1]))
            roll = 0.0
            cam_R_i = R_from_euler_xyz(pitch, yaw, roll)
            poses.append((cam_t_i, cam_R_i))
        return poses

    poses = generate_circular_control_sequence(cam_center, radius=CIRC_RADIUS, n_frames=N_FRAMES)
    ring_poly = ring_points3d(n=128)
    K = K_from_hfov(HFOV)

    frames_bgr = []
    masks = []
    bands = []
    contours_frames = []

    print("Rendering frames...")
    for t_idx, (ct, Rc) in enumerate(poses):
        frame = np.full((BASE_H, BASE_W, 3), 28, dtype=np.uint8)
        for i in range(N_RINGS):
            Rw, tw = ring_world_pose(i, STEP_M, VAR_DEG, seed=RANDOM_SEED)
            rvec, tvec = obj_to_cam(Rw, tw, Rc, ct)
            pts2d, _ = cv2.projectPoints(ring_poly, rvec, tvec, K, None)
            pts = np.round(pts2d).astype(np.int32).reshape(-1,1,2)
            col = (0, int(50 + 205 * (i / max(1, N_RINGS - 1))), 255)
            cv2.polylines(frame, [pts], True, col, thickness=10, lineType=cv2.LINE_AA)
        frame_noised = add_noise_blur(frame, noise_std=NOISE_STD, blur_k=BLUR_K if BLUR_K%2==1 else max(1, BLUR_K-1))
        contours, S, mask2, band = detect_rim_band(frame_noised)
        frames_bgr.append(frame_noised)
        masks.append(mask2.copy())
        bands.append(band.copy())
        contours_frames.append(contours)
        if (t_idx+1) % 30 == 0:
            print(f"  rendered frame {t_idx+1}/{N_FRAMES}")

    # sample detections
    pts_frames = []
    for t in range(len(contours_frames)):
        contours = contours_frames[t]
        pts_list = []
        for c in contours:
            if len(c) < MIN_CONTOUR_LEN: continue
            spts = sample_contour_points(c, n_pts=PTS_PER_CONTOUR)
            if spts.shape[0] > 0:
                pts_list.append(spts)
        if len(pts_list) == 0:
            pts_frames.append(np.zeros((0,2), np.float32))
        else:
            pts_frames.append(np.vstack(pts_list).astype(np.float32))

    # small debug video of sampled points
    pts_vid = []
    for t, pts in enumerate(pts_frames):
        img = frames_bgr[t].copy()
        if pts.shape[0] > 0:
            for p in pts.astype(np.int32):
                cv2.circle(img, tuple(p), 2, (0,255,0), -1)
        cv2.putText(img, f"Frame {t+1}/{len(frames_bgr)} pts={len(pts)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
        pts_vid.append(img)
    write_video(os.path.join(debug_dir, "01_sampled_points.mp4"), pts_vid, fps=30)

    # ellipse propagation (seed + extension)
    print("Ellipse propagation (seed + extension)...")
    individual_trajectories = ellipse_propagation_pipeline(frames_bgr, pts_frames, contours_frames, poses, K)
    print(f"Trajectories seeded and extended: {len(individual_trajectories)}")
    
    # ---------------- Trajectory Grouping (The Fix) ----------------
    print("Grouping individual trajectories into Super-Trajectories...")
    super_trajectories = group_trajectories_by_initial_fit(individual_trajectories, BASE_W, BASE_H)
    print(f"Grouped into {len(super_trajectories)} Super-Trajectories.")
    
    # Visualize the (many) individual trajectories colored by their future cluster ID
    n_traj = max(1, len(individual_trajectories))
    colors = color_list(max(16, n_traj))
    
    # Map the individual trajectories to their new Super-Trajectory label for visualization
    traj_label_map = {}
    for st in super_trajectories:
        for i in st['cluster_indices']:
            traj_label_map[i] = st['label']
            
    assigned_vid = []
    for t in range(len(frames_bgr)):
        img = frames_bgr[t].copy()
        for i, tr in enumerate(individual_trajectories):
            label = traj_label_map.get(i, -1)
            # Use the label index for coloring, or a generic color for noise (-1)
            col_idx = label % len(colors) if label != -1 else len(colors)-1
            col = tuple(int(c) for c in colors[col_idx])
            
            for (ft,x,y) in tr['trajectory']:
                if ft == t:
                    cv2.circle(img, (int(round(x)), int(round(y))), 3, col, -1)
                    break
        cv2.putText(img, f"Frame {t+1}/{len(frames_bgr)} (Points colored by Group ID)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
        assigned_vid.append(img)
    write_video(os.path.join(debug_dir, "02_tracked_points_by_group_ID.mp4"), assigned_vid, fps=30)
    
    # debug overlay of groups
    mid = len(frames_bgr)//2
    overlay = draw_trajectories_overlay(frames_bgr[mid], super_trajectories, colors, draw_points=True, draw_ellipses=False)
    cv2.imwrite(os.path.join(debug_dir, "03_trajectories_overlay.png"), overlay)

    # ---------------- depth-aware refinement (on Super-Trajectories) ----------------
    print("Refining ellipses using triangulation/parallax on Super-Trajectories...")
    refined_count = refine_ellipses_with_depth(super_trajectories, poses, K)
    print(f"Refined {refined_count} Super-Trajectories with 3D circle fit")

    # draw overlay with refined ellipses
    overlay_refined = draw_trajectories_overlay(frames_bgr[mid], super_trajectories, colors, draw_points=True, draw_ellipses=True)
    cv2.imwrite(os.path.join(debug_dir, "04_refined_ellipses_overlay.png"), overlay_refined)

    # build descriptors [cx,cy,w,h]
    descriptors_full = []
    traj_indices_full = []
    for i,tr in enumerate(super_trajectories):
        e = tr.get('refined_ellipse')
        if e is None:
            continue
        desc = ellipse_descriptor_from_params(e)
        descriptors_full.append(desc)
        traj_indices_full.append(i) # This is the index of the Super-Trajectory
        
    if len(descriptors_full) == 0:
        print("No descriptors available after grouping and refinement, exiting.")
        return
        
    descs_full = np.vstack(descriptors_full)

    # Filter abnormal descriptors (should be less needed now, but kept for safety)
    descs, traj_indices = filter_abnormal_descriptors(descs_full, traj_indices_full, BASE_W, BASE_H)

    if descs.shape[0] == 0:
        print("No valid descriptors remaining after filtering, exiting.")
        return
        
    descs_norm = normalize_descriptors(descs, W=BASE_W, H=BASE_H)

    # DBSCAN clustering (on normalized descriptors) - now for the final, clean set of rings
    eps_norm = DBSCAN_EPS / float((BASE_W + BASE_H) / 2.0)
    if SKLEARN_OK:
        # DBSCAN is now run on a small, clean set of points (approx 4)
        db = DBSCAN(eps=eps_norm, min_samples=DBSCAN_MIN_SAMPLES, metric='euclidean')
        labels = db.fit_predict(descs_norm)
    else:
        # fallback: simple kmeans (using N_RINGS=4)
        def kmeans_simple(X, k=N_RINGS):
            rng = np.random.default_rng(42)
            N = X.shape[0]
            k = min(k, N)
            if k == 0: return np.zeros(N, dtype=np.int64) - 1
            centers = X[rng.choice(N, size=k, replace=False)]
            for _ in range(50):
                d = np.linalg.norm(X[:,None,:] - centers[None,:,:], axis=2)
                labs = d.argmin(axis=1)
                newc = np.array([X[labs==i].mean(axis=0) if np.any(labs==i) else centers[i] for i in range(k)])
                if np.allclose(newc, centers): break
                centers = newc
            d = np.linalg.norm(X[:,None,:] - centers[None,:,:], axis=2)
            return d.argmin(axis=1)
            
        labels = kmeans_simple(descs_norm, N_RINGS)

    unique, counts = np.unique(labels, return_counts=True)
    print("Final Clustering result counts (Super-Trajectories):", dict(zip(unique, counts)))

    # save descriptors CSV
    with open(os.path.join(debug_dir, "ellipse_descriptors_labels.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["super_traj_index","cx","cy","w","h","label"])
        for idx, ti in enumerate(traj_indices):
            cx,cy,wid,hei = descs[idx]
            lab = int(labels[idx])
            w.writerow([int(ti), float(cx), float(cy), float(wid), float(hei), lab])

    # visualization: color all original points by their final cluster label
    label_colors = color_list(max(8, len(unique)))
    clustering_vid = []
    
    # Create a final map: Original Trajectory Index -> Final Cluster Label
    final_label_map = {}
    for st_idx, final_label in zip(traj_indices, labels):
        st = super_trajectories[st_idx]
        for original_idx in st['cluster_indices']:
            final_label_map[original_idx] = final_label
            
    # Now draw the individual points colored by the final cluster label
    for t in range(len(frames_bgr)):
        img = frames_bgr[t].copy()
        
        # Draw points for all original tracks
        for i, tr in enumerate(individual_trajectories):
            final_lab = final_label_map.get(i, -1) # -1 for noise/unrefined groups
            if final_lab == -1: continue # Don't draw points from unrefined groups
            
            col = tuple(int(c) for c in label_colors[final_lab % len(label_colors)])
            
            for (ft,x,y) in tr['trajectory']:
                if ft == t:
                    cv2.circle(img, (int(round(x)), int(round(y))), 3, col, -1)
                    break
                    
        # Also draw the refined ellipse on top
        for idx, st in enumerate(super_trajectories):
            if st.get('refined_ellipse') is not None and idx in traj_indices:
                final_lab = labels[traj_indices.index(idx)]
                col = tuple(int(c) for c in label_colors[final_lab % len(label_colors)])
                e = st['refined_ellipse']
                cv2.ellipse(img, ( (int(round(e[0][0])), int(round(e[0][1]))),
                                   (int(round(e[1][0])), int(round(e[1][1]))),
                                   float(e[2]) ), color=col, thickness=2, lineType=cv2.LINE_AA)

        cv2.putText(img, f"Frame {t+1}/{len(frames_bgr)} (Points colored by Final Cluster)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        clustering_vid.append(img)
        
    write_video(os.path.join(run_dir, "clustering_by_ellipse_descriptor.mp4"), clustering_vid, fps=30)

    # diagnostic scatter of centers colored by cluster
    plt.figure(figsize=(8,6))
    for lab in np.unique(labels):
        mask = (labels == lab)
        plt.scatter(descs[mask,0], descs[mask,1], label=f"lab {lab}", alpha=0.8)
    plt.xlabel("cx (px)"); plt.ylabel("cy (px)"); plt.title("Ellipse centers colored by cluster (Filtered Super-Trajectories)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, "05_descriptor_centers_by_cluster_refined.png"), dpi=150)
    plt.close()

    print("Outputs written:")
    print("  ", run_dir)
    print("  ", debug_dir)
    print("Done.")

if __name__ == "__main__":
    main()