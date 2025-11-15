import numpy as np
import cv2
import math

# ---------------- Config ----------------
RING_DIAM_M = 0.9144
RING_RAD_M  = RING_DIAM_M / 2.0

# ---------------- Camera / geometry helpers (kept for completeness) ----------------
# The simulation used the K_from_hfov internal function to get K.
# We will require K to be passed to the main function.
def R_from_euler_xyz(rx, ry, rz):
    """Generates a 3x3 rotation matrix from ZYX Euler angles (intrinsic)."""
    rx, ry, rz = map(math.radians, (rx, ry, rz))
    Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
    Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
    Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx

def rvec_from_R(R):
    """Converts a 3x3 rotation matrix to a 3-element rotation vector."""
    rvec, _ = cv2.Rodrigues(R)
    return rvec.astype(np.float64)

# ---------------- Detection ----------------
def detect_ellipses_fast(frame_bgr):
    """
    Detects ellipse candidates by finding contours on the saturation channel's gradient.
    Returns: list of ellipses (tuples: ((cx,cy),(MA,ma),ang)).
    """
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
        try: 
            e = cv2.fitEllipseAMS(c)
        except Exception:
            try: e = cv2.fitEllipse(c)
            except Exception: continue
            
        (cx,cy),(MA,ma),ang = e
        # Filter: minimum axis length and major axis > minor axis
        if min(MA,ma) < 12 or MA <= ma: continue
        ellipses.append(e)
        
    return ellipses

# ---------------- Core PnP Solver Implementation ----------------
def solve_ellipse_pnp_approx(K, ellipse, ring_radius_m):
    """
    Approximation for 3D Pose of a circle using the projected major axis for depth
    and the axis ratio for orientation.

    Inputs:
        K: 3x3 Camera Matrix
        ellipse: ((cx,cy),(MA,ma),ang) - The detected 2D ellipse parameters
        ring_radius_m: Radius of the 3D circle (hoop) in meters

    Returns: 
        (center_3d, normal_3d) or None.
        center_3d (np.array): [X_c, Y_c, Z_c] position in Camera Frame (meters).
        normal_3d (np.array): [N_x, N_y, N_z] unit normal vector in Camera Frame.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx0, cy0 = K[0, 2], K[1, 2]
    
    (cx, cy), (MA, ma), ang = ellipse
    
    # 1. Depth (Z_c) Calculation using Major Axis and Focal Length
    # Major Axis (MA) in pixels relates to real diameter (2*R) by: MA = fx * (2*R) / Z_c
    # Z_c = fx * (2*R) / MA
    diameter_world = 2.0 * ring_radius_m
    if MA < 1e-6: return None # Avoid division by zero
    
    Z_c = fx * diameter_world / MA
    
    # 2. Center Position (X_c, Y_c) Calculation (Inverse Perspective Projection)
    # X_c = (Z_c / fx) * (cx - cx0)
    # Y_c = (Z_c / fy) * (cy - cy0)
    X_c = (Z_c / fx) * (cx - cx0)
    Y_c = (Z_c / fy) * (cy - cy0)
    
    center_3d = np.array([X_c, Y_c, Z_c], dtype=np.float64)
    
    # 3. Normal Vector (N_c) Calculation (Approximation)
    # The slant angle (alpha) between the camera's Z-axis (line-of-sight) 
    # and the hoop's normal is given by the axis ratio: cos(alpha) = ma / MA
    # This assumes the hoop center is at the principal point, which is an approximation.
    ratio = min(1.0, ma / MA)
    alpha = math.acos(ratio) # Slant angle
    
    # Calculate the 3D rotation of the normal vector based on the ellipse's 2D orientation (ang)
    # and the slant angle (alpha).
    
    # Rotation 1: Yaw the normal towards the ellipse center (cx, cy)
    # This defines the plane in which the normal vector lies.
    R_yaw = R_from_euler_xyz(0, ang + 90, 0) # Rotate to minor axis direction

    # Rotation 2: Tilt (Pitch) the normal by the slant angle (alpha)
    R_tilt = R_from_euler_xyz(math.degrees(alpha), 0, 0)

    # Combined Rotation Matrix (approx)
    R_c = R_yaw @ R_tilt

    # Normal vector is the Z-axis of the hoop's local frame, transformed
    # We use the rotation defined by the major axis angle (ang) for the in-plane rotation
    # and the slant angle (alpha) for the tilt.
    
    # Start with the Z-axis (hoop normal when fully visible)
    N_local = np.array([0.0, 0.0, 1.0])
    
    # Rotate by the slant angle (alpha)
    # We rotate around the axis defined by the major axis angle (ang + 90)
    # in the image plane (which is the axis perpendicular to the line of sight and the normal).
    
    # The rotation matrix for the normal
    # We can simplify this by constructing a rotation from a normal vector
    
    # Simplest approximation: 
    # The vector in the image plane that the normal is tilted towards is along the minor axis.
    ang_rad = math.radians(ang)
    dir_x = math.sin(ang_rad) # Direction of minor axis (perpendicular to major axis)
    dir_y = -math.cos(ang_rad)
    
    # Create an orthonormal basis for the hoop's rotation (R_c)
    N_c = np.array([dir_x, dir_y, 0.0]) * math.sin(alpha)
    N_c[2] = math.cos(alpha)

    # Rotate N_c to align with the unprojected center vector (V_ctr) 
    # The normal should point AWAY from the camera center.
    V_ctr_norm = center_3d / (np.linalg.norm(center_3d) + 1e-9)
    
    # Construct an approximate rotation that aligns the plane defined by
    # the major axis with the slant angle:
    
    # Rz: rotation around Z-axis by ang
    Rz = np.array([[math.cos(math.radians(ang)), -math.sin(math.radians(ang)), 0],
                   [math.sin(math.radians(ang)), math.cos(math.radians(ang)), 0],
                   [0, 0, 1]])
    
    # Rx: rotation around the new X-axis by alpha
    Rx = np.array([[1, 0, 0], 
                   [0, math.cos(alpha), -math.sin(alpha)], 
                   [0, math.sin(alpha), math.cos(alpha)]])
    
    # R_tilt_angle = Rz @ Rx
    # This is a fixed rotation, but the normal direction should be along the minor axis.
    
    # The normal vector is a vector in the plane defined by the camera Z-axis and the minor axis direction.
    # A simplified normal (tilted by alpha in the minor axis direction, then normalized):
    
    normal_dir_2d = np.array([dir_x, dir_y])
    normal_dir_3d = np.append(normal_dir_2d * math.sin(alpha), math.cos(alpha))
    
    # Final step: If the hoop is viewed from "behind" (normal points towards camera), 
    # the depth is the same, but the normal should flip sign.
    # Since MA is always positive, this approximation *cannot* determine the sign flip.
    # We will assume the normal always points away from the camera.
    normal_3d = normal_dir_3d / np.linalg.norm(normal_dir_3d)

    return center_3d, normal_3d


# ---------------- Main Algorithm Function ----------------
def find_nearest_hoop_pose(bgr_image, K, ring_radius_m=RING_RAD_M):
    """
    Detects hoops in a single image and calculates the 3D pose of the nearest one.

    Args:
        bgr_image (np.array): Input image in BGR format.
        K (np.array): 3x3 Camera Intrinsic Matrix.
        ring_radius_m (float): Physical radius of the hoop in meters.

    Returns:
        tuple or None: (center_3d, normal_3d) of the nearest hoop, both in the camera frame.
                       center_3d (np.array): [X_c, Y_c, Z_c] position (meters).
                       normal_3d (np.array): [N_x, N_y, N_z] unit normal vector.
    """
    # 1. Detect 2D Ellipses
    ellipses = detect_ellipses_fast(bgr_image)

    if not ellipses:
        return None

    hoop_poses = []

    # 2. Estimate 3D Pose for each detected ellipse using the major axis approximation
    for ellipse in ellipses:
        # Call the PnP approximation solver
        pose_result = solve_ellipse_pnp_approx(K, ellipse, ring_radius_m)
        
        if pose_result is not None:
            center_3d, normal_3d = pose_result
            
            hoop_poses.append({
                'center_3d': center_3d,
                'normal_3d': normal_3d,
                'depth': center_3d[2] # Depth is Z_c
            })

    if not hoop_poses:
        return None

    # 3. Find the nearest hoop (minimum positive depth Z_c)
    valid_poses = [p for p in hoop_poses if p['depth'] > 0]
    if not valid_poses:
         return None

    nearest_hoop = min(valid_poses, key=lambda p: p['depth'])

    # 4. Return the results
    return nearest_hoop['center_3d'], nearest_hoop['normal_3d']

# The rest of the original simulation code is omitted as requested.
# To use this, you must call find_nearest_hoop_pose with a valid BGR image and K matrix.