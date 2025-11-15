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
# All world-level layout / randomness / spacing variables you might want to control.
# Change these values to control lateral placement, vertical band, spacing, etc.

# Visual / image
BASE_W, BASE_H = 960, 540

# Rings / spacing
N_RINGS = 4                
STEP_M = 0.9               # Reduced depth spacing for better separation
VAR_DEG = 12.0             

# Camera initial pose & motion
CAM_INIT = np.array([4.93, -3.20, -2.03], np.float64)
CAM_RX, CAM_RY, CAM_RZ = -16.3, -45.6, -2.1
HFOV = 78.0
CIRC_RADIUS = 3.5          # INCREASED from 2.5 to get better viewing angles
N_FRAMES = 180             

# Ring physical size
RING_DIAM_M = 0.9144
RING_RAD_M = RING_DIAM_M / 2.0

# Noise / blur applied to frames
NOISE_STD, BLUR_K = 7, 3

# World box geometry (for lateral fractions)
WORLD_SIZE = 8.0                 
LATERAL_FRACTIONS = (0.80, 0.20, 0.60, 0.40)  # 4 DISTINCT lateral positions

# Vertical / height control for rings
HEIGHT_MODE = 'jitter'     
VERTICAL_BAND = (2.5, 6.0)  # Expanded vertical range
VERTICAL_JITTER = 0.3       # Slightly larger jitter for more vertical separation

# Reproducible randomness
RANDOM_SEED = 42         

# Track filtering
MIN_TRACK_LIFETIME = 10

# Feature selection for clustering (based on analysis)
SELECTED_FEATURES = ['mag_mean', 'pos_std_y', 'dir_x']  # Most discriminative features

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
        target = np.array([0.0, 0.0, 4.0], np.float64)  # Adjusted target for better view
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

################################################################################
# Updated ring_world_pose: alternates lateral positions and samples heights in band
def ring_world_pose(i, step_m, var_deg,
                    lateral_fractions=LATERAL_FRACTIONS,
                    world_size=WORLD_SIZE,
                    height_mode=HEIGHT_MODE,
                    vertical_band=VERTICAL_BAND,
                    vertical_jitter=VERTICAL_JITTER,
                    seed=None):
    """
    Returns (R_w, t_w) for ring index i.

    Behavior:
      - lateral positions alternate among lateral_fractions (fractions of world width).
        fraction 0 => left edge (-world_size/2), fraction 1 => right edge (+world_size/2).
      - base height (z) is base_z + i * step_m (preserves lengthwise spacing).
      - If height_mode == 'jitter': z = base_z + i*step_m + uniform(-vertical_jitter, +vertical_jitter),
          then clamped to vertical_band.
      - If height_mode == 'uniform': z = uniform(vertical_band[0], vertical_band[1]).
    """
    if seed is not None:
        rng = np.random.default_rng(int(seed) + i)
    else:
        rng = np.random.default_rng()

    # lateral x-position: alternate fractions
    frac = lateral_fractions[i % len(lateral_fractions)]
    x = -world_size/2.0 + frac * world_size

    # lengthwise base and height handling (z is vertical)
    base_z = 3.0
    base_z_i = base_z + i * step_m

    min_z, max_z = vertical_band

    if height_mode == 'uniform':
        z_i = float(rng.uniform(min_z, max_z))
    else:  # 'jitter' or other -> keep base spacing + jitter clamped
        jitter = float(rng.uniform(-vertical_jitter, vertical_jitter))
        z_i = base_z_i + jitter
        z_i = float(np.clip(z_i, min_z, max_z))

    # lateral y (front/back) kept 0.0 by default
    y = 0.0

    t_w = np.array([x, y, z_i], np.float64)

    # small orientation variation kept as before
    rxi = var_deg * math.sin(0.7 * i)
    ryi = var_deg * math.cos(0.9 * i)
    rzi = var_deg * math.sin(1.3 * i)
    R_w = R_from_euler_xyz(rxi, ryi, rzi)
    return R_w, t_w
################################################################################

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
    """Enhanced feature extraction with better spatial discrimination"""
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
        
        # Basic motion features
        m_mean = mags.mean()
        m_std = mags.std()
        m_median = np.median(mags)
        u_mean = dirs.mean(axis=0)
        
        # Spatial features (normalized)
        p_mean = pts.mean(axis=0)
        p_std = pts.std(axis=0)
        p_mean_norm = p_mean / np.array([W, H], np.float32)
        p_std_norm = p_std / np.array([W, H], np.float32)
        
        # 10D feature
        aspect = p_std[0] / (p_std[1] + 1e-6)
        
        feats.append(np.array([
            m_mean, m_std, m_median,
            u_mean[0], u_mean[1],
            p_mean_norm[0], p_mean_norm[1],
            p_std_norm[0], p_std_norm[1],
            aspect
        ], np.float32))
    
    feats = np.vstack(feats)
    # Standardize
    mu = feats.mean(0)
    sig = feats.std(0) + 1e-6
    feats_norm = (feats - mu) / sig
    
    return feats_norm, feature_names, feats  # Return normalized, names, and raw

def extract_selected_features(feats_norm, feature_names, selected_names):
    """Extract only selected features from the full feature matrix"""
    indices = [feature_names.index(name) for name in selected_names]
    return feats_norm[:, indices]

def track_features_sequence(tracks, W, H):
    """Original 6D features for compatibility"""
    feats = []
    for tr in tracks:
        valid = ~np.isnan(tr).any(axis=1)
        pts = tr[valid]
        if pts.shape[0] < 2:
            feats.append(np.zeros(6, np.float32))
            continue
        d = np.diff(pts, axis=0)
        mags = np.linalg.norm(d, axis=1) + 1e-6
        dirs = d / mags[:,None]
        m_mean = mags.mean()
        m_std = mags.std()
        u_mean = dirs.mean(axis=0)
        p_mean = (pts.mean(axis=0) / np.array([W,H],np.float32))
        feats.append(np.array([m_mean, u_mean[0], u_mean[1], p_mean[0], p_mean[1], m_std], np.float32))
    feats = np.vstack(feats)
    mu = feats.mean(0)
    sig = feats.std(0) + 1e-6
    feats = (feats - mu) / sig
    return feats

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

def save_point_debug_videos(debug_dir, pts_for_frames, attr_for_frames, frame_imgs, attr_names, base_name):
    for j, attr_name in enumerate(attr_names):
        video = []
        for t, pts in enumerate(pts_for_frames):
            img = frame_imgs[t].copy()
            vals = attr_for_frames[j][t]
            cols = gradient_colormap(vals)
            for i,p in enumerate(pts.astype(int)):
                cv2.circle(img, tuple(p), 2, tuple(map(int,cols[i])), -1)
            video.append(img)
        write_video(os.path.join(debug_dir, f"{base_name}_{attr_name}.mp4"), video, fps=30)

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
    """Create interactive 3D plot of selected features"""
    print("\nCreating 3D feature space visualization...")
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(20, 5))
    
    # View 1: Colored by cluster label
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
    
    # View 2: Colored by contour ID
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
    
    # View 3: Different angle
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
    
    # View 4: Top-down
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
    
    # Create rotating animation
    print("Creating rotating 3D animation...")
    frames_3d = []
    for angle in range(0, 360, 3):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for label in np.unique(labels):
            mask = (labels == label)
            ax.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                      c=[colors_cluster[np.where(labels==label)[0][0]]/255.0],
                      label=f'Cluster {label}', s=30, alpha=0.7)
        
        ax.set_xlabel(feature_names[0], fontsize=12)
        ax.set_ylabel(feature_names[1], fontsize=12)
        ax.set_zlabel(feature_names[2], fontsize=12)
        ax.set_title(f'3D Feature Space (angle={angle}°)', fontsize=14)
        ax.view_init(elev=20, azim=angle)
        ax.legend()
        
        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frames_3d.append(img_bgr)
        plt.close()
    
    write_video(os.path.join(debug_dir, "3d_feature_space_rotating.mp4"), frames_3d, fps=20)
    print("✓ Saved 3D feature space visualizations")

def comprehensive_debug_diagnostics(debug_dir, tracks, frames_bgr, cids_frames, fb_errors, poses):
    """Comprehensive debugging suite to diagnose clustering issues"""
    N0 = len(tracks)
    T = len(frames_bgr)
    
    print("\n=== COMPREHENSIVE DIAGNOSTICS ===")
    
    # 1. Track lifetime analysis
    lifetimes = np.array([np.count_nonzero(~np.isnan(tr).any(axis=1)) for tr in tracks])
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(lifetimes, bins=30, edgecolor='black')
    plt.xlabel('Track lifetime (frames)')
    plt.ylabel('Count')
    plt.title(f'Track Lifetimes (mean={lifetimes.mean():.1f}, std={lifetimes.std():.1f})')
    plt.axvline(MIN_TRACK_LIFETIME, color='r', linestyle='--', label=f'Min threshold={MIN_TRACK_LIFETIME}')
    plt.legend()
    
    # 2. FB error distribution
    plt.subplot(1,2,2)
    all_fb = np.concatenate([np.array(fb) for fb in fb_errors if len(fb) > 0])
    if len(all_fb) > 0:
        plt.hist(all_fb, bins=50, range=(0,5), edgecolor='black')
        plt.xlabel('Forward-backward error (pixels)')
        plt.ylabel('Count')
        plt.title(f'FB Error Distribution (mean={all_fb.mean():.2f})')
        plt.axvline(1.5, color='r', linestyle='--', label='Threshold=1.5')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir,"01_track_quality.png"), dpi=150)
    plt.close()
    print(f"✓ Saved track quality analysis")
    
    # 3. Spatial distribution over time
    fig, axes = plt.subplots(2, 3, figsize=(15,10))
    sample_frames = [0, T//4, T//2, 3*T//4, T-1]
    for idx, t_idx in enumerate(sample_frames[:6]):
        ax = axes[idx//3, idx%3]
        pts = np.array([tracks[i][t_idx] for i in range(N0)])
        valid = ~np.isnan(pts).any(axis=1)
        if valid.any():
            colors = color_points_by(cids_frames[0] if len(cids_frames[0]) == N0 else np.arange(N0))
            ax.scatter(pts[valid,0], pts[valid,1], c=colors[valid]/255.0, s=10, alpha=0.7)
            ax.set_xlim(0, BASE_W)
            ax.set_ylim(BASE_H, 0)
            ax.set_title(f'Frame {t_idx}')
            ax.set_aspect('equal')
    plt.suptitle('Spatial Distribution of Tracks Over Time (colored by initial contour)')
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir,"02_spatial_evolution.png"), dpi=150)
    plt.close()
    print(f"✓ Saved spatial evolution")
    
    # 4. Final frame spatial scatter
    final_pts = np.array([tr[-1] for tr in tracks])
    valid = ~np.isnan(final_pts).any(axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    
    # By contour ID
    ax = axes[0]
    if len(cids_frames[0]) == N0:
        colors_contour = color_points_by(cids_frames[0])
        for cid in np.unique(cids_frames[0]):
            mask = (cids_frames[0] == cid) & valid
            if mask.any():
                ax.scatter(final_pts[mask,0], final_pts[mask,1], 
                          c=[colors_contour[np.where(cids_frames[0]==cid)[0][0]]/255.0],
                          s=30, label=f'Contour {cid}', alpha=0.7)
    ax.set_xlim(0, BASE_W)
    ax.set_ylim(BASE_H, 0)
    ax.set_title('Final Positions by Initial Contour ID')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    
    # Histogram of x-positions
    ax = axes[1]
    ax.hist(final_pts[valid,0], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('X position (pixels)')
    ax.set_ylabel('Count')
    ax.set_title(f'X-Position Distribution ({len(np.unique(cids_frames[0]))} contours detected)')
    # Mark expected positions
    for i, frac in enumerate(LATERAL_FRACTIONS):
        expected_x = BASE_W * frac
        ax.axvline(expected_x, color='r', linestyle='--', alpha=0.5, label=f'Expected ring {i+1}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir,"03_final_spatial_distribution.png"), dpi=150)
    plt.close()
    print(f"✓ Saved final spatial distribution")
    
    # 5. Feature analysis
    print("\nComputing feature vectors...")
    Feat10d_norm, feature_names, Feat10d_raw = track_features_enhanced(tracks, BASE_W, BASE_H)
    
    # Plot each feature dimension colored by contour
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (feat_name, feat_vals) in enumerate(zip(feature_names, Feat10d_norm.T)):
        ax = axes[i]
        if len(cids_frames[0]) == N0:
            for cid in np.unique(cids_frames[0]):
                mask = (cids_frames[0] == cid)
                ax.hist(feat_vals[mask], bins=20, alpha=0.6, label=f'Contour {cid}')
            ax.set_title(f'{feat_name}')
            ax.legend(fontsize=7)
        else:
            ax.hist(feat_vals, bins=20, alpha=0.7)
            ax.set_title(f'{feat_name}')
    
    for i in range(len(feature_names), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Feature Distributions by Contour (look for separation)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir,"04_feature_distributions.png"), dpi=150)
    plt.close()
    print(f"✓ Saved feature distributions")
    
    print(f"\n✓ Diagnostics complete")
    
    return Feat10d_norm, feature_names, Feat10d_raw

############### MAIN PIPELINE ###############
def main():
    run_dir = ensure_dir("algorithmOutputVisualizations/run_{:03d}".format(len(glob.glob("algorithmOutputVisualizations/run_*"))+1))
    debug_dir = ensure_dir("debugInfo/run_{:03d}".format(len(glob.glob("debugInfo/run_*"))+1))

    print(f"\n{'='*60}")
    print(f"STARTING PIPELINE - Run: {os.path.basename(run_dir)}")
    print(f"{'='*60}\n")
    print(f"Ring positions (x-coords): {[f'{-WORLD_SIZE/2 + f*WORLD_SIZE:.1f}m' for f in LATERAL_FRACTIONS]}")
    print(f"Camera orbit radius: {CIRC_RADIUS}m")
    print(f"Selected features for clustering: {SELECTED_FEATURES}\n")

    # Sequence generation
    print("Generating camera trajectory and poses...")
    poses = generate_circular_control_sequence(CAM_INIT.copy(), radius=CIRC_RADIUS, n_frames=N_FRAMES)
    ring_poly = ring_points3d(n=128)
    K, fx, fy, cx, cy = K_from_hfov(HFOV)
    
    print(f"Rendering {N_FRAMES} frames with {N_RINGS} rings...")
    frames_bgr, masks, bands, pts_frames, cids_frames, contours_frames = [], [], [], [], [], []
    
    for t_idx, (ct, Rc) in enumerate(poses):
        if t_idx % 30 == 0:
            print(f"  Frame {t_idx}/{N_FRAMES}...")
        K, fx, fy, cx, cy = K_from_hfov(HFOV)
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
        
        pts_this, cids_this = [], []
        for cid, c in enumerate(contours):
            if len(c)<20: continue
            pts = sample_contour_points(c, n_pts=60)
            if pts.shape[0]==0: continue
            pts_this.append(pts)
            cids_this.extend([cid]*len(pts))
        if len(pts_this)==0:
            pts_frames.append(np.zeros((0,2),np.float32))
            cids_frames.append(np.zeros((0,),np.int32))
        else:
            pts_frames.append(np.vstack(pts_this))
            cids_frames.append(np.array(cids_this,np.int32))

    print(f"✓ Rendered {len(frames_bgr)} frames")
    print(f"✓ Detected {len(np.unique(cids_frames[0]))} unique contours in first frame")
    
    # Save basic videos
    print("\nSaving basic detection videos...")
    save_mask_and_band_videos(debug_dir, masks, bands, frames_bgr)

    video_points_by_contour = []
    for t, pts in enumerate(pts_frames):
        img = frames_bgr[t].copy()
        cols = color_points_by(cids_frames[t])
        for i,p in enumerate(pts.astype(int)):
            cv2.circle(img, tuple(p), 2, tuple(map(int,cols[i])), -1)
        video_points_by_contour.append(img)
    write_video(os.path.join(debug_dir,"sampled_points_by_contour.mp4"), video_points_by_contour, fps=30)
    print("✓ Saved sampled points video")

    # Tracking
    print("\nRunning optical flow tracking...")
    init_points = pts_frames[0]
    tracks, fb_errors = compute_tracks_lk(frames_bgr, init_points, fb_thresh=1.5)
    N0 = len(init_points)
    T = len(frames_bgr)
    print(f"✓ Tracked {N0} points across {T} frames")
    
    track_frames_by_id = []
    for t in range(T):
        img = frames_bgr[t].copy()
        for i,p in enumerate([tracks[j][t] for j in range(N0)]):
            if not np.any(np.isnan(p)):
                col = tuple(int(x) for x in color_points_by([i])[0])
                cv2.circle(img, tuple(p.astype(int)), 2, col, -1)
        track_frames_by_id.append(img)
    write_video(os.path.join(debug_dir,"tracks_by_id.mp4"), track_frames_by_id, fps=30)

    # COMPREHENSIVE DIAGNOSTICS
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE DIAGNOSTICS")
    print("="*60)
    Feat10d_norm, feature_names, Feat10d_raw = comprehensive_debug_diagnostics(
        debug_dir, tracks, frames_bgr, cids_frames, fb_errors, poses)

    # Filter short tracks
    print(f"\nFiltering tracks (min lifetime = {MIN_TRACK_LIFETIME} frames)...")
    lifetimes = np.array([np.count_nonzero(~np.isnan(tr).any(axis=1)) for tr in tracks])
    good_idx = np.where(lifetimes >= MIN_TRACK_LIFETIME)[0]
    print(f"  Kept {len(good_idx)} / {N0} tracks ({100*len(good_idx)/N0:.1f}%)")
    
    tracks_filtered = [tracks[i] for i in good_idx]
    cids_filtered = cids_frames[0][good_idx] if len(cids_frames[0]) == N0 else np.arange(len(good_idx))

    # Compute features on filtered tracks
    print("\nComputing features on filtered tracks...")
    Feat10d_norm_filt, feature_names, Feat10d_raw_filt = track_features_enhanced(tracks_filtered, BASE_W, BASE_H)
    
    # Extract selected features
    print(f"Extracting selected features: {SELECTED_FEATURES}")
    Feat3d_selected = extract_selected_features(Feat10d_norm_filt, feature_names, SELECTED_FEATURES)
    
    # Cluster using selected features
    print(f"\nClustering using {len(SELECTED_FEATURES)} selected features...")
    labels_selected, centers = kmeans_nd(Feat3d_selected, N_RINGS, iters=50, seed=42)
    
    # Compute clustering quality
    if SKLEARN_OK and len(tracks_filtered) > N_RINGS:
        try:
            sil = silhouette_score(Feat3d_selected, labels_selected)
            db = davies_bouldin_score(Feat3d_selected, labels_selected)
            print(f"  Silhouette score: {sil:.3f} (higher is better)")
            print(f"  Davies-Bouldin index: {db:.3f} (lower is better)")
        except Exception as e:
            print(f"  Could not compute metrics: {e}")
    
    # Cluster size distribution
    unique, counts = np.unique(labels_selected, return_counts=True)
    print(f"  Cluster sizes: {dict(zip(unique, counts))}")
    
    # Create 3D feature space visualization
    plot_3d_feature_space(debug_dir, Feat3d_selected, labels_selected, cids_filtered, SELECTED_FEATURES)
    
    # Generate video with selected feature clustering
    print("\nGenerating 'newFeatureUsage' video with selected features...")
    video_new_features = []
    for t in range(T):
        img = frames_bgr[t].copy()
        for i in range(len(tracks_filtered)):
            p = tracks_filtered[i][t]
            if not np.any(np.isnan(p)):
                col = tuple(int(x) for x in color_points_by(labels_selected)[i])
                cv2.circle(img, tuple(p.astype(int)), 3, col, -1)
        
        # Add text overlay
        cv2.putText(img, f"Features: {', '.join(SELECTED_FEATURES)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(img, f"Frame {t}/{T}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        video_new_features.append(img)
    
    write_video(os.path.join(run_dir, "newFeatureUsage_clustering.mp4"), video_new_features, fps=30)
    print("✓ Saved newFeatureUsage video")
    
    # Save CSV with selected feature clustering results
    save_csv(
        [(i, labels_selected[i], cids_filtered[i]) for i in range(len(tracks_filtered))],
        ["track_id", "cluster_label", "initial_contour_id"],
        os.path.join(debug_dir, f"selected_features_clustering.csv")
    )
    
    # Also test other feature combinations for comparison
    print("\nTesting other feature combinations for comparison...")
    cluster_attr_sets = {
        "full_10d": Feat10d_norm_filt,
        "selected_3d": Feat3d_selected,
        "spatial_only": extract_selected_features(Feat10d_norm_filt, feature_names, ['pos_x', 'pos_y']),
    }
    
    comparison_results = []
    for k, featset in cluster_attr_sets.items():
        labels, _ = kmeans_nd(featset, N_RINGS, iters=50, seed=42)
        
        video = []
        for t in range(T):
            img = frames_bgr[t].copy()
            for i in range(len(tracks_filtered)):
                p = tracks_filtered[i][t]
                if not np.any(np.isnan(p)):
                    col = tuple(int(x) for x in color_points_by(labels)[i])
                    cv2.circle(img, tuple(p.astype(int)), 2, col, -1)
            video.append(img)
        write_video(os.path.join(debug_dir, f"clusters_by_{k}.mp4"), video, fps=30)
        
        # Compute metrics
        if SKLEARN_OK:
            try:
                sil = silhouette_score(featset, labels)
                db = davies_bouldin_score(featset, labels)
                comparison_results.append((k, sil, db))
                print(f"  {k}: Silhouette={sil:.3f}, DB={db:.3f}")
            except:
                pass
    
    # Save comparison
    with open(os.path.join(debug_dir, "feature_comparison.txt"), 'w') as f:
        f.write("FEATURE SET COMPARISON\n")
        f.write("="*50 + "\n\n")
        for name, sil, db in comparison_results:
            f.write(f"{name}:\n")
            f.write(f"  Silhouette: {sil:.3f}\n")
            f.write(f"  Davies-Bouldin: {db:.3f}\n\n")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"Results saved to:")
    print(f"  Visualizations: {run_dir}")
    print(f"  Debug info: {debug_dir}")
    print(f"  3D feature plot: {debug_dir}/06_3d_feature_space.png")
    print(f"  3D rotating video: {debug_dir}/3d_feature_space_rotating.mp4")
    print(f"  Main output: {run_dir}/newFeatureUsage_clustering.mp4")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()