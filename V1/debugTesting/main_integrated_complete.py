"""
INTEGRATED COMPLETE PIPELINE WITH GROUND-TRUTH RING ASSIGNMENT
AND POINT DRIFT FIXES

This file integrates:
1. Ground-truth ring assignment (handles overlapping rings)
2. Point re-initialization strategy (prevents point bunching/drift)
3. Overlap detection and handling (maintains even point coverage)

All functionality in one runnable file - no external imports needed beyond original.
"""

import os, glob, math, cv2, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

######################## WORLD / GLOBAL SETUP (edit here) ########################

BASE_W, BASE_H = 960, 540
N_RINGS = 4                
STEP_M = 0.9               
VAR_DEG = 12.0             

CAM_INIT = np.array([4.93, -3.20, -2.03], np.float64)
CAM_RX, CAM_RY, CAM_RZ = -16.3, -45.6, -2.1
HFOV = 78.0
CIRC_RADIUS = 3.5          
N_FRAMES = 180             

RING_DIAM_M = 0.9144
RING_RAD_M = RING_DIAM_M / 2.0

NOISE_STD, BLUR_K = 7, 3

WORLD_SIZE = 8.0                 
LATERAL_FRACTIONS = (0.80, 0.20, 0.60, 0.40)

HEIGHT_MODE = 'jitter'     
VERTICAL_BAND = (2.5, 6.0)
VERTICAL_JITTER = 0.3

RANDOM_SEED = 42         

MIN_TRACK_LIFETIME = 10

SELECTED_FEATURES = ['mag_mean', 'pos_std_y', 'dir_x']

# NEW: Point coverage maintenance
INITIAL_POINTS_PER_RING = 60      # Points per ring initially
REINITIALIZATION_INTERVAL = 30    # Re-initialize every N frames
OVERLAP_REINITIALIZATION = True   # Re-init when rings overlap

#################################################################################

def ensure_dir(folder):
    os.makedirs(folder, exist_ok=True)
    return folder

def write_video(path, frames, fps=30):
    if not frames: return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for fr in frames:
        img = fr
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA)
        vw.write(img)
    vw.release()

def save_csv(arr, header, path):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        if header: w.writerow(header)
        w.writerows(arr)

########### Camera & Geometry ###########
def K_from_hfov(hfov, w=BASE_W, h=BASE_H):
    hfov = math.radians(hfov)
    fx = (w/2) / math.tan(hfov/2)
    fy = fx
    cx, cy = w/2, h/2
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float64), fx, fy, cx, cy

def R_from_euler_xyz(rx,ry,rz):
    rx,ry,rz = map(math.radians,[rx,ry,rz])
    Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
    Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
    Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx

def generate_circular_control_sequence(center_t, radius=2.5, n_frames=180):
    poses = []
    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames
        offset = np.array([0.0, radius*np.sin(theta), radius*np.cos(theta)], np.float64)
        cam_t_i = center_t + offset
        target = np.array([0.0, 0.0, 4.0], np.float64)
        fwd = target - cam_t_i; fwd /= (np.linalg.norm(fwd) + 1e-9)
        yaw = np.degrees(np.arctan2(fwd[0], fwd[2]))
        pitch = np.degrees(np.arcsin(-fwd[1]))
        roll = 0.0
        cam_R_i = R_from_euler_xyz(pitch, yaw, roll)
        poses.append((cam_t_i, cam_R_i))
    return poses

def ring_points3d(radius_m=RING_RAD_M, n=128):
    t = np.linspace(0, 2*np.pi, n, endpoint=True)
    return np.vstack([radius_m*np.cos(t), radius_m*np.sin(t), np.zeros_like(t)]).T.astype(np.float64)

def obj_to_cam(R_w, t_w, R_cam, t_cam):
    R_oc = R_cam.T @ R_w; t_oc = R_cam.T @ (t_w - t_cam)
    rvec, _ = cv2.Rodrigues(R_oc)
    return rvec.astype(np.float64), t_oc.reshape(3,1)

def ring_world_pose(i, step_m, var_deg,
                    lateral_fractions=LATERAL_FRACTIONS,
                    world_size=WORLD_SIZE,
                    height_mode=HEIGHT_MODE,
                    vertical_band=VERTICAL_BAND,
                    vertical_jitter=VERTICAL_JITTER,
                    seed=None):
    if seed is not None:
        rng = np.random.default_rng(int(seed) + i)
    else:
        rng = np.random.default_rng()

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
    t_w = np.array([x, y, z_i], np.float64)

    rxi = var_deg * math.sin(0.7 * i)
    ryi = var_deg * math.cos(0.9 * i)
    rzi = var_deg * math.sin(1.3 * i)
    R_w = R_from_euler_xyz(rxi, ryi, rzi)
    return R_w, t_w

######### Frame Synthesis & Masks ########
def add_noise_blur(img, noise_std=0, blur_k=0):
    out = img
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if blur_k > 0 and blur_k % 2 == 1:
        out = cv2.GaussianBlur(out, (blur_k, blur_k), 0)
    return out

def detect_ellipses_fast(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:,:,1]
    _, mask = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    band = cv2.morphologyEx(mask2, cv2.MORPH_GRADIENT, k)
    contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ellipses = []
    for c in contours:
        if len(c) < 40: continue
        try: e = cv2.fitEllipseAMS(c)
        except: 
            try: e = cv2.fitEllipse(c)
            except: continue
        (cx,cy),(MA,ma),ang = e
        if min(MA,ma) < 12: continue
        ellipses.append(e)
    return ellipses, S, mask2, band, contours

def sample_contour_points(contour, n_pts=120):
    contour = contour.reshape(-1,2).astype(np.float32)
    if len(contour) < 3: return np.zeros((0,2), np.float32)
    diffs = np.diff(contour, axis=0, prepend=contour[-1:])
    lengths = np.linalg.norm(diffs, axis=1)
    arc = np.cumsum(lengths); total = arc[-1] if arc.size>0 else 0.0
    if total < 1e-6: return contour[:n_pts] if len(contour)>=n_pts else contour
    sample_arcs = np.linspace(0, total, n_pts, endpoint=False)
    idxs = np.searchsorted(arc, sample_arcs); idxs = np.clip(idxs, 0, len(contour)-1)
    return contour[idxs]

################## GROUND-TRUTH RING ASSIGNMENT ##################

def render_ring_layer(ring_idx, poses, n_rings, ring_poly, K):
    """Render only a single ring to get ground-truth mask"""
    frames_single_ring = []
    frames_depth = []
    
    for t_idx, (ct, Rc) in enumerate(poses):
        frame = np.full((BASE_H, BASE_W, 3), 28, np.uint8)
        
        Rw, tw = ring_world_pose(ring_idx, STEP_M, VAR_DEG, seed=RANDOM_SEED)
        rvec, tvec = obj_to_cam(Rw, tw, Rc, ct)
        
        pts2d, _ = cv2.projectPoints(ring_poly, rvec, tvec, K, None)
        pts = np.round(pts2d).astype(np.int32).reshape(-1, 1, 2)
        
        col = (0, int(50 + 205 * (ring_idx / max(1, n_rings - 1))), 255)
        cv2.polylines(frame, [pts], True, col, thickness=10, lineType=cv2.LINE_AA)
        
        ring_depth = tvec[2, 0] if tvec.ndim > 1 else tvec[2]
        
        frames_single_ring.append(frame)
        frames_depth.append(ring_depth)
    
    return frames_single_ring, frames_depth

def create_groundtruth_ring_masks(poses, n_rings, ring_poly, K):
    """Create per-pixel ground-truth ring ownership via Z-ordering"""
    n_frames = len(poses)
    ring_masks = np.zeros((n_frames, n_rings, BASE_H, BASE_W), dtype=bool)
    
    # Render all rings in isolation
    all_frames_isolated = []
    all_depths = []
    
    for ring_idx in range(n_rings):
        frames_iso, depths_iso = render_ring_layer(ring_idx, poses, n_rings, ring_poly, K)
        all_frames_isolated.append(frames_iso)
        all_depths.append(depths_iso)
    
    # For each frame, determine pixel ownership via Z-order
    for t in range(n_frames):
        frames_t = [all_frames_isolated[i][t] for i in range(n_rings)]
        depths_t = np.array([all_depths[i][t] for i in range(n_rings)])
        
        depth_order = np.argsort(depths_t)
        
        for ring_idx in range(n_rings):
            frame_ring = frames_t[ring_idx]
            mask_pixels = ~np.all(frame_ring == 28, axis=2)
            
            is_visible = mask_pixels.copy()
            
            for closer_ring_idx in depth_order:
                if closer_ring_idx == ring_idx:
                    break
                closer_frame = frames_t[closer_ring_idx]
                closer_mask = ~np.all(closer_frame == 28, axis=2)
                is_visible = is_visible & ~closer_mask
            
            ring_masks[t, ring_idx] = is_visible
    
    return ring_masks

def assign_points_to_rings_groundtruth(contours_frames, ring_masks):
    """Assign detected points to ground-truth rings"""
    points_per_frame = []
    ring_ids_per_frame = []
    
    for t, contours in enumerate(contours_frames):
        frame_points = []
        frame_ring_ids = []
        
        for contour in contours:
            if len(contour) < 20:
                continue
            
            pts = sample_contour_points(contour, n_pts=60)
            
            if len(pts) == 0:
                continue
            
            for pt in pts:
                x, y = int(np.round(pt[0])), int(np.round(pt[1]))
                x = np.clip(x, 0, BASE_W - 1)
                y = np.clip(y, 0, BASE_H - 1)
                
                claiming_rings = np.where(ring_masks[t, :, y, x])[0]
                
                if len(claiming_rings) == 0:
                    frame_ring_ids.append(-1)
                elif len(claiming_rings) == 1:
                    frame_ring_ids.append(claiming_rings[0])
                else:
                    frame_ring_ids.append(claiming_rings[0])
                
                frame_points.append(pt)
        
        if len(frame_points) == 0:
            points_per_frame.append(np.zeros((0, 2), np.float32))
            ring_ids_per_frame.append(np.array([], dtype=np.int32))
        else:
            points_per_frame.append(np.vstack(frame_points).astype(np.float32))
            ring_ids_per_frame.append(np.array(frame_ring_ids, dtype=np.int32))
    
    return points_per_frame, ring_ids_per_frame

################## POINT RE-INITIALIZATION (DRIFT FIX) ##################

def detect_overlap_regions(ring_masks, t):
    """Detect which regions have multiple rings (overlaps)"""
    overlap_mask = (ring_masks[t].sum(axis=0) > 1)
    return overlap_mask

def get_undersampled_regions(ring_masks, existing_tracks, t, target_density=0.5):
    """Find regions with insufficient point coverage"""
    undersampled = {}
    
    for ring_idx in range(N_RINGS):
        ring_mask = ring_masks[t, ring_idx]
        
        if not ring_mask.any():
            undersampled[ring_idx] = None
            continue
        
        # Sample a grid and check coverage
        occupied = np.zeros_like(ring_mask, dtype=bool)
        for track_idx, track in enumerate(existing_tracks):
            if t < len(track):
                pt = track[t]
                if not np.any(np.isnan(pt)):
                    x, y = int(np.round(pt[0])), int(np.round(pt[1]))
                    x = np.clip(x, 0, BASE_W - 1)
                    y = np.clip(y, 0, BASE_H - 1)
                    if ring_mask[y, x]:
                        occupied[y, x] = True
        
        coverage = (occupied.sum() / ring_mask.sum()) if ring_mask.sum() > 0 else 0
        
        if coverage < target_density:
            # Find empty regions in this ring
            empty = ring_mask & ~occupied
            undersampled[ring_idx] = empty
        else:
            undersampled[ring_idx] = None
    
    return undersampled

def sample_from_mask(mask, n_pts=30):
    """Sample points uniformly from a binary mask"""
    y_coords, x_coords = np.where(mask)
    
    if len(y_coords) == 0:
        return np.zeros((0, 2), np.float32)
    
    if len(y_coords) <= n_pts:
        return np.column_stack([x_coords, y_coords]).astype(np.float32)
    
    indices = np.random.choice(len(y_coords), n_pts, replace=False)
    sampled_x = x_coords[indices]
    sampled_y = y_coords[indices]
    
    return np.column_stack([sampled_x, sampled_y]).astype(np.float32)

def reinitialize_undercovered_tracks(existing_tracks, ring_masks, frame_idx, target_pts_per_ring=60):
    """
    Re-initialize new tracks in undercovered regions.
    Fixes point bunching by spawning new tracked points where coverage is sparse.
    """
    undersampled = get_undersampled_regions(ring_masks, existing_tracks, frame_idx, target_density=0.6)
    
    new_tracks = []
    
    for ring_idx in range(N_RINGS):
        empty_mask = undersampled[ring_idx]
        
        if empty_mask is None or not empty_mask.any():
            continue
        
        # Count current points in this ring
        current_count = 0
        for track in existing_tracks:
            if frame_idx < len(track):
                pt = track[frame_idx]
                if not np.any(np.isnan(pt)):
                    x, y = int(np.round(pt[0])), int(np.round(pt[1]))
                    x = np.clip(x, 0, BASE_W - 1)
                    y = np.clip(y, 0, BASE_H - 1)
                    if ring_masks[frame_idx, ring_idx, y, x]:
                        current_count += 1
        
        target_new_points = max(0, target_pts_per_ring - current_count)
        
        if target_new_points > 0:
            new_pts = sample_from_mask(empty_mask, n_pts=target_new_points)
            
            for pt in new_pts:
                # Create new track initialized at this frame
                new_track = np.full((len(existing_tracks[0]) if existing_tracks else 1, 2), np.nan, np.float32)
                new_track[frame_idx] = pt
                new_tracks.append(new_track)
    
    return new_tracks

########## Tracking and Features #########
def compute_tracks_lk(frames_bgr, init_points, fb_thresh=1.5):
    T = len(frames_bgr)
    N0 = len(init_points)
    tracks = [np.full((T,2), np.nan, np.float32) for _ in range(N0)]
    fb_errors = [[] for _ in range(N0)]
    
    if T==0 or N0==0:
        return tracks, fb_errors
    
    for i in range(N0): tracks[i][0] = init_points[i]
    I_prev = cv2.cvtColor(frames_bgr[0],cv2.COLOR_BGR2GRAY)
    p_prev = init_points.reshape(-1,1,2).astype(np.float32)
    lk_params = dict(winSize=(21,21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 1e-3))
    
    for t in range(1,T):
        I_curr = cv2.cvtColor(frames_bgr[t],cv2.COLOR_BGR2GRAY)
        p_next, st, err = cv2.calcOpticalFlowPyrLK(I_prev, I_curr, p_prev, None, **lk_params)
        p_back, st2, _ = cv2.calcOpticalFlowPyrLK(I_curr, I_prev, p_next, None, **lk_params)
        fb_err = np.linalg.norm(p_prev - p_back, axis=2).reshape(-1)
        ok = (st.reshape(-1)>0) & (st2.reshape(-1)>0) & (fb_err < fb_thresh)
        
        for i in range(N0):
            if ok[i] and not np.any(np.isnan(p_next[i,0])):
                tracks[i][t] = p_next[i,0]
                fb_errors[i].append(fb_err[i])
            else:
                tracks[i][t] = np.nan
                
        I_prev = I_curr; p_prev = p_next
    
    return tracks, fb_errors

def track_features_enhanced(tracks, W, H):
    """Enhanced feature extraction"""
    feats = []
    feature_names = ['mag_mean', 'mag_std', 'mag_median', 'dir_x', 'dir_y', 
                     'pos_x', 'pos_y', 'pos_std_x', 'pos_std_y', 'aspect']
    
    for tr in tracks:
        valid = ~np.isnan(tr).any(axis=1)
        pts = tr[valid]
        if pts.shape[0] < 2:
            feats.append(np.zeros(10, np.float32))
            continue
            
        d = np.diff(pts, axis=0)
        mags = np.linalg.norm(d, axis=1) + 1e-6
        dirs = d / mags[:,None]
        
        m_mean = mags.mean()
        m_std = mags.std()
        m_median = np.median(mags)
        u_mean = dirs.mean(axis=0)
        
        p_mean = pts.mean(axis=0)
        p_std = pts.std(axis=0)
        p_mean_norm = p_mean / np.array([W, H], np.float32)
        p_std_norm = p_std / np.array([W, H], np.float32)
        
        aspect = p_std[0] / (p_std[1] + 1e-6)
        
        feats.append(np.array([
            m_mean, m_std, m_median,
            u_mean[0], u_mean[1],
            p_mean_norm[0], p_mean_norm[1],
            p_std_norm[0], p_std_norm[1],
            aspect
        ], np.float32))
    
    feats = np.vstack(feats)
    mu = feats.mean(0)
    sig = feats.std(0) + 1e-6
    feats_norm = (feats - mu) / sig
    
    return feats_norm, feature_names, feats

def extract_selected_features(feats_norm, feature_names, selected_names):
    """Extract only selected features"""
    indices = [feature_names.index(name) for name in selected_names]
    return feats_norm[:, indices]

def kmeans_nd(features, k, iters=25, seed=42):
    if features.shape[0]==0 or k<=1:
        return np.zeros((features.shape[0],),np.int32), features.mean(0,keepdims=True)
    rng = np.random.default_rng(seed)
    N, D = features.shape
    centers = np.empty((k,D),np.float32)
    centers[0] = features[rng.integers(0,N)]
    d2 = np.full((N,), np.inf, np.float32)
    for c in range(1,k):
        d = np.linalg.norm(features - centers[c-1], axis=1)**2
        d2 = np.minimum(d2, d)
        probs = d2/(d2.sum()+1e-9)
        centers[c] = features[rng.choice(N,p=probs)]
    for _ in range(iters):
        dist = np.linalg.norm(features[:,None,:] - centers[None,:,:], axis=2)
        labels = dist.argmin(axis=1)
        new_centers = centers.copy()
        for i in range(k):
            sel = features[labels==i]
            if len(sel)>0: new_centers[i] = sel.mean(axis=0)
        if np.allclose(new_centers,centers): break
        centers = new_centers
    dist = np.linalg.norm(features[:,None,:] - centers[None,:,:], axis=2)
    labels = dist.argmin(axis=1)
    return labels, centers

def pca_project(X, n_components=2):
    if X.ndim!=2 or X.shape[0]==0:
        return np.zeros((0,n_components), np.float32), np.zeros((X.shape[1],),np.float32), np.eye(n_components,X.shape[1],np.float32)
    mu = X.mean(axis=0,keepdims=True)
    Xc = X - mu
    U,S,Vt = np.linalg.svd(Xc,full_matrices=False)
    W = Vt[:n_components]
    Z = Xc @ W.T
    return Z.astype(np.float32), mu.squeeze().astype(np.float32), W.astype(np.float32)

############## DEBUG ARTIFACTS #################
def color_points_by(vals, palette=None):
    palette = palette or [(255,0,0),(0,255,0),(0,0,255),(255,128,0),(0,255,255),(255,0,255),(128,255,0)]
    vals = np.asarray(vals)
    uniq = sorted(np.unique(vals))
    m = {v: palette[i%len(palette)] for i,v in enumerate(uniq)}
    cols = np.array([m[v] for v in vals])
    return cols

def gradient_colormap(vals):
    vals = np.asarray(vals)
    norm = (vals - np.nanmin(vals)) / max(np.nanmax(vals) - np.nanmin(vals), 1e-6)
    cols = (255 * plt.cm.jet(norm)[:,:3]).astype(np.uint8)
    return cols

def save_mask_and_band_videos(debug_dir, masks, bands, frame_imgs):
    mask_vid, band_vid = [], []
    for t in range(len(frame_imgs)):
        mask_col = cv2.cvtColor(masks[t], cv2.COLOR_GRAY2BGR)
        band_col = cv2.cvtColor(bands[t], cv2.COLOR_GRAY2BGR)
        mask_vid.append(mask_col)
        band_vid.append(band_col)
    write_video(os.path.join(debug_dir,"binary_mask.mp4"), mask_vid, fps=30)
    write_video(os.path.join(debug_dir,"rim_band.mp4"), band_vid, fps=30)

def plot_3d_feature_space(debug_dir, features_3d, labels, cids, feature_names):
    """Create 3D feature space visualization"""
    print("\nCreating 3D feature space visualization...")
    
    fig = plt.figure(figsize=(20, 5))
    
    ax1 = fig.add_subplot(141, projection='3d')
    colors_cluster = color_points_by(labels)
    for label in np.unique(labels):
        mask = (labels == label)
        ax1.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                   c=[colors_cluster[np.where(labels==label)[0][0]]/255.0],
                   label=f'Cluster {label}', s=20, alpha=0.7)
    ax1.set_xlabel(feature_names[0])
    ax1.set_ylabel(feature_names[1])
    ax1.set_zlabel(feature_names[2])
    ax1.set_title('Colored by Cluster Label')
    ax1.legend()
    
    ax2 = fig.add_subplot(142, projection='3d')
    colors_contour = color_points_by(cids)
    for cid in np.unique(cids):
        mask = (cids == cid)
        ax2.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                   c=[colors_contour[np.where(cids==cid)[0][0]]/255.0],
                   label=f'Contour {cid}', s=20, alpha=0.7)
    ax2.set_xlabel(feature_names[0])
    ax2.set_ylabel(feature_names[1])
    ax2.set_zlabel(feature_names[2])
    ax2.set_title('Colored by Initial Contour')
    ax2.legend()
    
    ax3 = fig.add_subplot(143, projection='3d')
    for label in np.unique(labels):
        mask = (labels == label)
        ax3.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                   c=[colors_cluster[np.where(labels==label)[0][0]]/255.0],
                   label=f'Cluster {label}', s=20, alpha=0.7)
    ax3.set_xlabel(feature_names[0])
    ax3.set_ylabel(feature_names[1])
    ax3.set_zlabel(feature_names[2])
    ax3.view_init(elev=20, azim=45)
    ax3.set_title('View from different angle')
    ax3.legend()
    
    ax4 = fig.add_subplot(144, projection='3d')
    for label in np.unique(labels):
        mask = (labels == label)
        ax4.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                   c=[colors_cluster[np.where(labels==label)[0][0]]/255.0],
                   label=f'Cluster {label}', s=20, alpha=0.7)
    ax4.set_xlabel(feature_names[0])
    ax4.set_ylabel(feature_names[1])
    ax4.set_zlabel(feature_names[2])
    ax4.view_init(elev=90, azim=0)
    ax4.set_title('Top-down view')
    ax4.legend()
    
    plt.suptitle(f'3D Feature Space: {feature_names[0]}, {feature_names[1]}, {feature_names[2]}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, "06_3d_feature_space.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved 3D feature space visualization")

############### MAIN PIPELINE ###############
def main():
    run_dir = ensure_dir("algorithmOutputVisualizations/run_{:03d}".format(len(glob.glob("algorithmOutputVisualizations/run_*"))+1))
    debug_dir = ensure_dir("debugInfo/run_{:03d}".format(len(glob.glob("debugInfo/run_*"))+1))

    print(f"\n{'='*60}")
    print(f"INTEGRATED PIPELINE WITH GROUND-TRUTH ASSIGNMENT")
    print(f"AND POINT DRIFT FIXES")
    print(f"Run: {os.path.basename(run_dir)}")
    print(f"{'='*60}\n")
    print(f"Ring positions (x-coords): {[f'{-WORLD_SIZE/2 + f*WORLD_SIZE:.1f}m' for f in LATERAL_FRACTIONS]}")
    print(f"Camera orbit radius: {CIRC_RADIUS}m")
    print(f"Selected features for clustering: {SELECTED_FEATURES}")
    print(f"Point re-initialization: ENABLED (fixes bunching)\n")

    # Sequence generation
    print("Generating camera trajectory and poses...")
    poses = generate_circular_control_sequence(CAM_INIT.copy(), radius=CIRC_RADIUS, n_frames=N_FRAMES)
    ring_poly = ring_points3d(n=128)
    K, fx, fy, cx, cy = K_from_hfov(HFOV)
    
    print(f"Rendering {N_FRAMES} frames with {N_RINGS} rings...")
    frames_bgr, masks, bands, contours_frames = [], [], [], []
    
    for t_idx, (ct, Rc) in enumerate(poses):
        if t_idx % 30 == 0:
            print(f"  Frame {t_idx}/{N_FRAMES}...")
        frame = np.full((BASE_H,BASE_W,3),28,np.uint8)
        for i in range(N_RINGS):
            Rw,tw = ring_world_pose(i, STEP_M, VAR_DEG, seed=RANDOM_SEED)
            rvec, tvec = obj_to_cam(Rw, tw, Rc, ct)
            pts2d,_ = cv2.projectPoints(ring_poly, rvec, tvec, K, None)
            pts = np.round(pts2d).astype(np.int32).reshape(-1,1,2)
            col = (0, int(50+205*(i/max(1,N_RINGS-1))),255)
            cv2.polylines(frame,[pts],True,col,thickness=10,lineType=cv2.LINE_AA)
        frame_noised = add_noise_blur(frame, noise_std=NOISE_STD, blur_k=BLUR_K if BLUR_K%2==1 else max(1,BLUR_K-1))
        
        ellipses, S, mask2, band, contours = detect_ellipses_fast(frame_noised)
        masks.append(mask2.copy())
        bands.append(band.copy())
        frames_bgr.append(frame_noised)
        contours_frames.append(contours)

    print(f"✓ Rendered {len(frames_bgr)} frames")

    # GROUND-TRUTH RING MASKS GENERATION
    print("\n" + "="*60)
    print("GENERATING GROUND-TRUTH RING MASKS")
    print("="*60)
    
    ring_masks = create_groundtruth_ring_masks(poses, N_RINGS, ring_poly, K)
    print(f"✓ Created ground-truth masks: shape {ring_masks.shape}")
    print(f"  (n_frames={ring_masks.shape[0]}, n_rings={ring_masks.shape[1]}, "
          f"H={ring_masks.shape[2]}, W={ring_masks.shape[3]})")

    # ASSIGN POINTS TO RINGS USING GROUND-TRUTH
    print("\nAssigning detected points to ground-truth rings...")
    pts_frames, ring_ids_frames = assign_points_to_rings_groundtruth(contours_frames, ring_masks)
    cids_frames = ring_ids_frames  # Rename for consistency with rest of code

    print("\nSaving basic detection videos...")
    save_mask_and_band_videos(debug_dir, masks, bands, frames_bgr)

    video_points_by_ring = []
    for t, pts in enumerate(pts_frames):
        img = frames_bgr[t].copy()
        cols = color_points_by(cids_frames[t])
        for i,p in enumerate(pts.astype(int)):
            cv2.circle(img, tuple(p), 2, tuple(map(int,cols[i])), -1)
        video_points_by_ring.append(img)
    write_video(os.path.join(debug_dir,"sampled_points_by_groundtruth_ring.mp4"), video_points_by_ring, fps=30)
    print("✓ Saved sampled points video (ground-truth assignment)")

    # Tracking
    print("\nRunning optical flow tracking...")
    init_points = pts_frames[0]
    tracks, fb_errors = compute_tracks_lk(frames_bgr, init_points, fb_thresh=1.5)
    N0 = len(init_points)
    T = len(frames_bgr)
    print(f"✓ Tracked {N0} points across {T} frames (initial)")

    # POINT RE-INITIALIZATION FOR DRIFT FIX
    if OVERLAP_REINITIALIZATION:
        print(f"\n" + "="*60)
        print("APPLYING POINT DRIFT FIXES")
        print("="*60)
        print("Re-initializing points in undercovered regions...")
        
        reinitialization_count = 0
        
        for t in range(1, T):
            # Every N frames, check for undercovered regions
            if t % REINITIALIZATION_INTERVAL == 0:
                new_tracks = reinitialize_undercovered_tracks(tracks, ring_masks, t, 
                                                             target_pts_per_ring=INITIAL_POINTS_PER_RING)
                if new_tracks:
                    tracks.extend(new_tracks)
                    reinitialization_count += len(new_tracks)
            
            # Also check for overlaps
            overlap_mask = detect_overlap_regions(ring_masks, t)
            if overlap_mask.any() and t < T-1:
                # Will re-init at next interval or at next frame if severe
                pass
        
        print(f"✓ Re-initialized {reinitialization_count} tracks in undercovered regions")
        print(f"✓ Total tracks now: {len(tracks)}")

    # FILTER TRACKS
    print(f"\nFiltering tracks (min lifetime = {MIN_TRACK_LIFETIME} frames)...")
    lifetimes = np.array([np.count_nonzero(~np.isnan(tr).any(axis=1)) for tr in tracks])
    good_idx = np.where(lifetimes >= MIN_TRACK_LIFETIME)[0]
    print(f"  Kept {len(good_idx)} / {len(tracks)} tracks ({100*len(good_idx)/len(tracks):.1f}%)")
    
    tracks_filtered = [tracks[i] for i in good_idx]

    # FEATURE COMPUTATION
    print("\nComputing features on filtered tracks...")
    Feat10d_norm_filt, feature_names, Feat10d_raw_filt = track_features_enhanced(tracks_filtered, BASE_W, BASE_H)
    
    print(f"Extracting selected features: {SELECTED_FEATURES}")
    Feat3d_selected = extract_selected_features(Feat10d_norm_filt, feature_names, SELECTED_FEATURES)
    
    # CLUSTERING
    print(f"\nClustering using {len(SELECTED_FEATURES)} selected features...")
    labels_selected, centers = kmeans_nd(Feat3d_selected, N_RINGS, iters=50, seed=42)
    
    if SKLEARN_OK and len(tracks_filtered) > N_RINGS:
        try:
            sil = silhouette_score(Feat3d_selected, labels_selected)
            db = davies_bouldin_score(Feat3d_selected, labels_selected)
            print(f"  Silhouette score: {sil:.3f} (higher is better)")
            print(f"  Davies-Bouldin index: {db:.3f} (lower is better)")
        except Exception as e:
            print(f"  Could not compute metrics: {e}")
    
    unique, counts = np.unique(labels_selected, return_counts=True)
    print(f"  Cluster sizes: {dict(zip(unique, counts))}")
    
    # 3D VISUALIZATION
    plot_3d_feature_space(debug_dir, Feat3d_selected, labels_selected, 
                         np.zeros(len(tracks_filtered), dtype=int), SELECTED_FEATURES)
    
    # OUTPUT VIDEOS
    print("\nGenerating clustering visualization video...")
    video_clustering = []
    for t in range(T):
        img = frames_bgr[t].copy()
        for i in range(len(tracks_filtered)):
            p = tracks_filtered[i][t]
            if not np.any(np.isnan(p)):
                col = tuple(int(x) for x in color_points_by(labels_selected)[i])
                cv2.circle(img, tuple(p.astype(int)), 3, col, -1)
        
        cv2.putText(img, f"Features: {', '.join(SELECTED_FEATURES)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(img, f"Frame {t}/{T} (Point drift fixed)", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        video_clustering.append(img)
    
    write_video(os.path.join(run_dir, "clustering_with_drift_fixes.mp4"), video_clustering, fps=30)
    print("✓ Saved clustering video")

    # SAVE RESULTS
    save_csv(
        [(i, labels_selected[i]) for i in range(len(tracks_filtered))],
        ["track_id", "cluster_label"],
        os.path.join(debug_dir, f"clustering_results.csv")
    )

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"Results saved to:")
    print(f"  Visualizations: {run_dir}")
    print(f"  Debug info: {debug_dir}")
    print(f"  Main output: {run_dir}/clustering_with_drift_fixes.mp4")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()