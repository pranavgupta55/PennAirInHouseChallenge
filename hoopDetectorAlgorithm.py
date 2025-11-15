import numpy as np
import cv2
import math
from typing import Optional, Tuple, List, Dict, Any, Union
import os
from collections import deque
import random

# --- Configuration ---
RING_RAD_M_DEFAULT = 0.9144 / 2.0  # Basketball hoop radius
BASE_W, BASE_H = 960, 540
HFOV_DEG_DEFAULT = 78.0
TEMPORAL_MAX_DIST = 150.0

# --- RELAXED Detection Parameters for Debugging ---
MIN_CONTOUR_POINTS = 30  # Reduced from 50
MIN_CONTOUR_AREA = 25    # Reduced from 100
MIN_AXIS_LENGTH = 5      # Reduced from 10
MAX_AXIS_LENGTH = 1500   # Add upper bound
ELLIPSE_FIT_ERROR_THRESHOLD = 10.0  # DECREASED from 5.0 for stricter validation (as requested)
RANSAC_ITERATIONS = 300  # INCREASED from 50 to check more combinations (as requested)
RANSAC_MIN_POINTS = 5    # Reduced
RANSAC_SAMPLE_POINTS = 10
RANSAC_INLIER_THRESHOLD = 7.0  # Increased tolerance

# --- State Class for Temporal Tracking ---
class HoopTrackerState:
    """Stores the last known state of the tracked hoop."""
    def __init__(self):
        self.last_center_2d: Optional[np.ndarray] = None
        self.last_depth: Optional[float] = None
        self.last_normal: Optional[np.ndarray] = None
        self.velocity_2d: Optional[np.ndarray] = None
        self.detection_history = deque(maxlen=5)
        self.lost_frames = 0
        
    def update(self, center_3d: np.ndarray, ellipse: tuple):
        """Updates state with new detection."""
        new_center_2d = np.array([ellipse[0][0], ellipse[0][1]])
        
        if self.last_center_2d is not None:
            if self.velocity_2d is None:
                self.velocity_2d = new_center_2d - self.last_center_2d
            else:
                new_velocity = new_center_2d - self.last_center_2d
                self.velocity_2d = 0.3 * new_velocity + 0.7 * self.velocity_2d
        
        self.last_center_2d = new_center_2d
        self.last_depth = center_3d[2]
        self.detection_history.append({
            'center_2d': new_center_2d.copy(),
            'depth': center_3d[2],
            'ellipse': ellipse
        })
        self.lost_frames = 0
        
    def predict_next_position(self):
        """Predicts next position."""
        if self.last_center_2d is not None and self.velocity_2d is not None:
            return self.last_center_2d + self.velocity_2d
        return self.last_center_2d
    
    def mark_lost(self):
        """Marks detection as lost."""
        self.lost_frames += 1
        if self.lost_frames > 10:
            self.last_center_2d = None
            self.velocity_2d = None

# --- Geometry Helpers ---
def get_default_K(W=BASE_W, H=BASE_H, hfov_deg=HFOV_DEG_DEFAULT):
    """Calculates K matrix."""
    hfov = math.radians(hfov_deg)
    fx = (W/2) / math.tan(hfov/2)
    fy = fx
    cx, cy = W/2, H/2
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float64)

# --- Simple Ellipse Fitting with Validation ---
def fit_ellipse_with_validation(points):
    """
    Try multiple ellipse fitting methods and return the best one.
    """
    if len(points) < 5:
        return None, float('inf'), "too_few_points"
    
    best_ellipse = None
    best_error = float('inf')
    best_method = None
    
    # Method 1: Standard fitEllipse
    try:
        ellipse = cv2.fitEllipse(points)
        error = calculate_ellipse_error_simple(points, ellipse)
        if error < best_error:
            best_error = error
            best_ellipse = ellipse
            best_method = "fitEllipse"
    except:
        pass
    
    # Method 2: fitEllipseAMS (more robust)
    try:
        ellipse = cv2.fitEllipseAMS(points)
        error = calculate_ellipse_error_simple(points, ellipse)
        if error < best_error:
            best_error = error
            best_ellipse = ellipse
            best_method = "fitEllipseAMS"
    except:
        pass
    
    # Method 3: fitEllipseDirect (least squares)
    try:
        ellipse = cv2.fitEllipseDirect(points)
        error = calculate_ellipse_error_simple(points, ellipse)
        if error < best_error:
            best_error = error
            best_ellipse = ellipse
            best_method = "fitEllipseDirect"
    except:
        pass
    
    return best_ellipse, best_error, best_method

def calculate_ellipse_error_simple(points, ellipse):
    """Simplified error calculation."""
    (cx, cy), (MA, ma), angle = ellipse
    
    # Basic validation
    if MA < 1e-6 or ma < 1e-6:
        return float('inf')
    
    # Sample some points on the ellipse and find nearest distances
    num_samples = min(36, len(points))
    angles = np.linspace(0, 2*np.pi, num_samples)
    
    # Generate ellipse points
    a, b = MA/2, ma/2
    ellipse_points = []
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    for theta in angles:
        x = a * np.cos(theta)
        y = b * np.sin(theta)
        # Rotate
        x_rot = x * cos_a - y * sin_a + cx
        y_rot = x * sin_a + y * cos_a + cy
        ellipse_points.append([x_rot, y_rot])
    
    ellipse_points = np.array(ellipse_points)
    
    # For each contour point, find distance to nearest ellipse point
    total_error = 0
    for pt in points:
        distances = np.linalg.norm(ellipse_points - pt, axis=1)
        total_error += np.min(distances)
    
    return total_error / len(points)

# --- SIMPLIFIED Detection using Saturation Channel ---
def detect_ellipses_fast_tuned(frame_bgr, debug=False) -> Tuple[List, Dict[str, np.ndarray]]:
    """
    Simplified detection focusing on saturation channel with better debugging.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:,:,1]
    
    # Visualization of saturation
    S_vis = cv2.cvtColor(S, cv2.COLOR_GRAY2BGR)
    S_vis = cv2.convertScaleAbs(S_vis, alpha=255.0/np.max(S)) if np.max(S) > 0 else S_vis
    
    # Binary threshold
    _, mask = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    
    # Create rim band (gradient)
    k_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    band = cv2.morphologyEx(mask_closed, cv2.MORPH_GRADIENT, k_grad)
    
    # Find contours
    contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Debug: visualize all contours found
    contour_debug = frame_bgr.copy()
    cv2.drawContours(contour_debug, contours, -1, (255, 255, 0), 1)
    cv2.putText(contour_debug, f"Found {len(contours)} contours", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Visualize sampled points
    sampled_points_vis = frame_bgr.copy()
    
    ellipses = []
    overlay_candidates = frame_bgr.copy()
    
    rejection_reasons = {
        'too_small': 0,
        'too_few_points': 0,
        'fit_failed': 0,
        'bad_axes': 0,
        'high_error': 0,
        'accepted': 0
    }
    
    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        
        # Very relaxed filtering for debugging
        if len(c) < MIN_CONTOUR_POINTS:
            rejection_reasons['too_few_points'] += 1
            continue
            
        if area < MIN_CONTOUR_AREA:
            rejection_reasons['too_small'] += 1
            continue
        
        # Convert contour to points
        points = c.squeeze()
        if points.ndim != 2:
            rejection_reasons['fit_failed'] += 1
            continue
        
        # Sample points from contour for visualization
        num_samples = min(20, len(points))
        sample_indices = np.linspace(0, len(points)-1, num_samples, dtype=int)
        sampled = points[sample_indices]
        
        # Draw sampled points
        for pt in sampled:
            cv2.circle(sampled_points_vis, tuple(pt), 3, (0, 255, 0), -1)
        
        # Try to fit ellipse
        ellipse, error, method = fit_ellipse_with_validation(points)
        
        if ellipse is None:
            rejection_reasons['fit_failed'] += 1
            # Draw failed contour in red
            cv2.drawContours(overlay_candidates, [c], -1, (0, 0, 255), 1)
            continue
        
        (cx, cy), (MA, ma), ang = ellipse
        
        # Check axes
        if MA < MIN_AXIS_LENGTH or ma < MIN_AXIS_LENGTH:
            rejection_reasons['bad_axes'] += 1
            # Draw in orange
            cv2.ellipse(overlay_candidates, ellipse, (0, 128, 255), 1, cv2.LINE_AA)
            continue
            
        if MA > MAX_AXIS_LENGTH or ma > MAX_AXIS_LENGTH:
            rejection_reasons['bad_axes'] += 1
            continue
            

        # # ------------------------------------------------------------------
        # # VITAL CHANGE: Relaxing the strict MA > ma filter for robustness
        # # ------------------------------------------------------------------
        # # if MA <= ma:  # Major axis should be larger (Original strict filter)
        # #     rejection_reasons['bad_axes'] += 1
        # #     continue
        
        # # New: Only reject if the fit is numerically an inverse ellipse (which should not happen)
        # # or if MA is only marginally larger than ma, allowing for near-circles due to viewing angle/noise
        # if MA < ma * 0.99: # E.g., Major axis must be at least 5% larger than Minor axis
        #    rejection_reasons['bad_axes'] += 1
        #    continue
        # # If you still have issues, you could comment out the entire MA <= ma section.
        # # ------------------------------------------------------------------

        
        # Check error
        if error > ELLIPSE_FIT_ERROR_THRESHOLD:
            rejection_reasons['high_error'] += 1
            # Draw in yellow
            cv2.ellipse(overlay_candidates, ellipse, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(overlay_candidates, f"e={error:.1f}", (int(cx-20), int(cy)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            continue
        
        # Accepted!
        rejection_reasons['accepted'] += 1
        ellipses.append(ellipse)
        
        # Draw accepted ellipse in green
        cv2.ellipse(overlay_candidates, ellipse, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay_candidates, f"{method}:{error:.1f}", (int(cx-20), int(cy)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw the sampled points that were accepted
        for pt in sampled:
            cv2.circle(sampled_points_vis, tuple(pt), 4, (0, 0, 255), -1)
    
    # Add rejection stats to overlay
    y_pos = 60
    for reason, count in rejection_reasons.items():
        text = f"{reason}: {count}"
        cv2.putText(overlay_candidates, text, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
    
    intermediate_frames = {
        '01_Saturation_Channel': S_vis,
        '02_Binary_Mask': cv2.cvtColor(mask_closed, cv2.COLOR_GRAY2BGR),
        '03_Rim_Band': cv2.cvtColor(band, cv2.COLOR_GRAY2BGR),
        '04_Ellipse_Candidates': overlay_candidates,
        '05_All_Contours': contour_debug,
        '06_Sampled_Points': sampled_points_vis
    }
    
    return ellipses, intermediate_frames

# --- PnP with Multiple Sizes ---
def solve_ellipse_pnp_approx(K, ellipse, ring_radius_m) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Simple PnP for single ring size."""
    fx, fy = K[0, 0], K[1, 1]
    cx0, cy0 = K[0, 2], K[1, 2]
    (cx, cy), (MA, ma), ang = ellipse
    diameter_world = 2.0 * ring_radius_m
    
    if MA < 1e-6:
        return None
    
    Z_c = fx * diameter_world / MA
    if Z_c < 0.1 or Z_c > 50.0:  # Reasonable range
        return None
    
    X_c = (Z_c / fx) * (cx - cx0)
    Y_c = (Z_c / fy) * (cy - cy0)
    center_3d = np.array([X_c, Y_c, Z_c], dtype=np.float64)
    
    # Normal approximation
    ratio = min(1.0, ma / MA)
    alpha = math.acos(ratio)
    ang_rad = math.radians(ang + 90)
    dir_x = math.cos(ang_rad)
    dir_y = math.sin(ang_rad)
    normal_dir_3d = np.array([dir_x, dir_y, 0.0]) * math.sin(alpha)
    normal_dir_3d[2] = math.cos(alpha)
    normal_3d = normal_dir_3d / np.linalg.norm(normal_dir_3d)
    
    return center_3d, normal_3d

# --- Main Detection Function ---
def find_nearest_hoop_pose(bgr_image, K, ring_radius_m, tracker_state: HoopTrackerState):
    """
    Main detection function with simplified approach.
    """
    ellipses, intermediate_frames = detect_ellipses_fast_tuned(bgr_image, debug=True)
    
    # Add detection count to first frame
    detection_status = bgr_image.copy()
    cv2.putText(detection_status, f"Detected {len(ellipses)} ellipse(s)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if len(ellipses) > 0 else (0, 0, 255), 2)
    intermediate_frames['00_Detection_Status'] = detection_status
    
    if not ellipses:
        tracker_state.mark_lost()
        return None, intermediate_frames
    
    # Try each ellipse with each radius size
    hoop_candidates = []
    
    # If ring_radius_m is a list, try all sizes
    if isinstance(ring_radius_m, list):
        radii_to_try = ring_radius_m
    else:
        radii_to_try = [ring_radius_m]
    
    for ellipse in ellipses:
        for radius in radii_to_try:
            pose_result = solve_ellipse_pnp_approx(K, ellipse, radius)
            if pose_result is not None:
                center_3d, normal_3d = pose_result
                hoop_candidates.append({
                    'center_3d': center_3d,
                    'normal_3d': normal_3d,
                    'depth': center_3d[2],
                    'ellipse': ellipse,
                    'center_2d': np.array([ellipse[0][0], ellipse[0][1]]),
                    'used_radius': radius
                })
    
    if not hoop_candidates:
        tracker_state.mark_lost()
        return None, intermediate_frames
    
    # Choose nearest
    nearest_hoop = min(hoop_candidates, key=lambda p: p['depth'])
    
    # Update tracker
    tracker_state.update(nearest_hoop['center_3d'], nearest_hoop['ellipse'])
    
    return (nearest_hoop['center_3d'], nearest_hoop['normal_3d'], 
            nearest_hoop['ellipse'], nearest_hoop['used_radius'], intermediate_frames)

# --- Visualization ---
def draw_hoop_pose_overlay(frame, K, center_3d, normal_3d, ellipse, used_radius=None):
    """Draw detection overlay."""
    cv2.ellipse(frame, ellipse, (0, 255, 0), 2, cv2.LINE_AA)
    
    cx, cy = int(round(ellipse[0][0])), int(round(ellipse[0][1]))
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    
    # Draw normal
    NORMAL_LINE_LENGTH = 0.5
    P_end_3d = center_3d + NORMAL_LINE_LENGTH * normal_3d
    
    rvec_zero = np.zeros((3,1), dtype=np.float64)
    tvec_zero = np.zeros((3,1), dtype=np.float64)
    points_3d = np.vstack([center_3d, P_end_3d]).reshape(-1, 1, 3)
    
    pts2d, _ = cv2.projectPoints(points_3d, rvec_zero, tvec_zero, K, None)
    
    p_center_2d = tuple(np.int32(pts2d[0, 0]))
    p_end_2d = tuple(np.int32(pts2d[1, 0]))
    
    cv2.line(frame, p_center_2d, p_end_2d, (255, 0, 0), 2, cv2.LINE_AA)