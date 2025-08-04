# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from glob import glob

# --- CONFIGURATION ---
input_folders = [
    "/home/AD.UNLV.EDU/bhattb3/Plot_Labeling/2024-07-19_Renamed/",
    "/home/AD.UNLV.EDU/bhattb3/Plot_Labeling/2024-08-16_Renamed/",
    "/home/AD.UNLV.EDU/bhattb3/Plot_Labeling/2024-09-19_Renamed/",
    # Add more folders if needed
]
output_folder = "/home/AD.UNLV.EDU/bhattb3/snow_melt/temporal_harris_fixed2/"
os.makedirs(output_folder, exist_ok=True)

RGB_SIZE = (8000, 6000)
THERMAL_SIZE = (640, 512)

plot_numbers = [i for i in range(1, 23) if i not in [14, 18]]

plot_colors = {
    1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 255, 0), 4: (0, 255, 255),
    5: (255, 0, 255), 6: (255, 255, 0), 7: (128, 0, 128), 8: (0, 128, 255),
    9: (128, 255, 0), 10: (255, 128, 0), 11: (0, 128, 128), 12: (128, 0, 0),
    13: (0, 0, 128), 15: (64, 64, 64), 16: (192, 0, 192), 17: (0, 192, 192),
    19: (100, 100, 0), 20: (0, 100, 100), 21: (150, 150, 50), 22: (50, 150, 150)
}

# --- FUNCTIONS ---
def get_image_type(filename):
    img = cv2.imread(filename)
    if img is None: return "UNKNOWN"
    h, w = img.shape[:2]
    if (w, h) == RGB_SIZE:
        return "RGB"
    elif (w, h) == THERMAL_SIZE:
        return "THERMAL"
    else:
        return "RGB" if w > 1000 else "THERMAL"


'''
def auto_align_images(ref_img, img_to_align):
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_img, None)

    if des1 is None or des2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 10:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)

    if H is None or mask is None or mask.sum() < 20:
        print("   Homography rejected: insufficient inliers.")
        return None

    aligned_img = cv2.warpPerspective(img_to_align, H, (ref_img.shape[1], ref_img.shape[0]))
    return aligned_img



def auto_align_images(ref_img, img_to_align, max_corner_shift=150, min_inliers=8):
    """
    Aligns img_to_align to ref_img using ORB features and affine transform.
    Rejects alignment if corner displacement too large or too few inliers.

    Parameters:
        ref_img: Reference BGR image.
        img_to_align: Image to align.
        max_corner_shift: Max allowed pixel shift for any corner after alignment.
                          Increase this (e.g. 150-300) for large images and more tolerance.
        min_inliers: Minimum number of inliers from RANSAC for accepting transform.
                     Set according to feature match quality, 8 is a reasonable start.

    Returns:
        Aligned image (np.ndarray) or None if alignment rejected.
    """
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_img, None)

    if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < min_inliers:
        return None

    matches = sorted(matches, key=lambda x: x.distance)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:30]])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:30]])

    M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=5)
    if M is None or (inliers is not None and np.sum(inliers) < min_inliers):
        return None

    h, w = ref_img.shape[:2]
    corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
    warped_corners = cv2.transform(corners, M)
    shift = np.linalg.norm(warped_corners - corners, axis=2).max()

    if shift > max_corner_shift:
        print(f"Alignment rejected: corner shift {shift:.1f} px > max {max_corner_shift}")
        return None

    aligned_img = cv2.warpAffine(img_to_align, M, (w, h))
    return aligned_img



def auto_align_images(ref_img, img_to_align, max_corner_shift=300, min_inliers=8, max_features=5000, ransac_thresh=5):
    """
    Align img_to_align to ref_img using ORB features and affine transform with multiple quality checks.

    Parameters:
        ref_img (np.ndarray): Reference BGR image.
        img_to_align (np.ndarray): Image to align.
        max_corner_shift (float): Max pixel corner displacement allowed after alignment.
                                  Increase for larger images or more tolerant alignment.
        min_inliers (int): Minimum RANSAC inlier matches required for accepting the transform.
        max_features (int): Number of ORB features to detect.
        ransac_thresh (float): RANSAC reprojection threshold.

    Returns:
        np.ndarray or None: Aligned image if quality checks pass; else None.
    """

    # Convert images to grayscale
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_img, None)

    # Check for valid descriptors
    if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
        print("Insufficient keypoints/descriptors detected.")
        return None

    # Match descriptors using BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Check if enough matches
    if len(matches) < min_inliers:
        print(f"Insufficient matches: {len(matches)} found, {min_inliers} required.")
        return None

    # Sort matches by distance (best first)
    matches = sorted(matches, key=lambda x: x.distance)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:max_features]])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:max_features]])

    # Estimate affine transform using RANSAC
    M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)

    # Check for valid transform and inlier count
    if M is None:
        print("Failed to estimate affine transform.")
        return None

    if inliers is not None and np.sum(inliers) < min_inliers:
        print(f"Too few inliers after RANSAC: {np.sum(inliers)} < {min_inliers}")
        return None

    # Calculate corner displacements to check for extreme transforms
    h, w = ref_img.shape[:2]
    corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
    warped_corners = cv2.transform(corners, M)
    max_shift = np.linalg.norm(warped_corners - corners, axis=2).max()

    if max_shift > max_corner_shift:
        print(f"Alignment rejected: corner shift {max_shift:.1f} px > max {max_corner_shift}")
        return None

    # Apply affine warp to align the image
    aligned_img = cv2.warpAffine(img_to_align, M, (w, h))
    return aligned_img
'''

def auto_align_images(
    ref_img,
    img_to_align,
    max_corner_shift=640,
    min_inliers=5,
    max_features=5000,
    ransac_thresh=5,
    downsample_scale=1.0
):
    """
    Align img_to_align to ref_img using ORB features and affine transform with quality checks.

    Params:
        ref_img (np.ndarray): Reference BGR image.
        img_to_align (np.ndarray): Image to align.
        max_corner_shift (float): Max allowed pixel corner shift after alignment (in full res pixels).
        min_inliers (int): Minimum RANSAC inliers to accept alignment.
        max_features (int): Max ORB features to detect.
        ransac_thresh (float): RANSAC reprojection threshold (pixels).
        downsample_scale (float): Rescale factor for alignment (<=1.0). Downsample for more stable matching.

    Returns:
        np.ndarray or None: Aligned image or None if rejected.
    """
    # Optionally downsample images to improve stability and speed
    if downsample_scale < 1.0:
        interp = cv2.INTER_AREA if downsample_scale < 1.0 else cv2.INTER_LINEAR
        ref_ds = cv2.resize(ref_img, None, fx=downsample_scale, fy=downsample_scale, interpolation=interp)
        align_ds = cv2.resize(img_to_align, None, fx=downsample_scale, fy=downsample_scale, interpolation=interp)
    else:
        ref_ds = ref_img
        align_ds = img_to_align

    gray_ref = cv2.cvtColor(ref_ds, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(align_ds, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_img, None)

    if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
        print("Insufficient keypoints or descriptors detected.")
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < min_inliers:
        print(f"Insufficient matches: {len(matches)} found, {min_inliers} required.")
        return None

    matches = sorted(matches, key=lambda x: x.distance)
    max_pts = min(max_features, len(matches))
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:max_pts]])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:max_pts]])

    M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    if M is None:
        print("Failed to estimate affine transform.")
        return None

    if inliers is not None and np.sum(inliers) < min_inliers:
        print(f"Too few inliers after RANSAC: {np.sum(inliers)} < {min_inliers}")
        return None

    h_ref, w_ref = ref_ds.shape[:2]
    corners = np.float32([[0,0], [w_ref,0], [w_ref,h_ref], [0,h_ref]]).reshape(-1,1,2)
    warped_corners = cv2.transform(corners, M)
    max_shift_ds = np.linalg.norm(warped_corners - corners, axis=2).max()

    # Scale max shift back to original size if downsampling is used
    max_shift = max_shift_ds / downsample_scale

    if max_shift > max_corner_shift:
        print(f"Alignment rejected: corner shift {max_shift:.1f} px > max {max_corner_shift}")
        return None

    # Warp original high-res image by scaled affine matrix
    if downsample_scale < 1.0:
        # Scale affine matrix to full resolution
        M_full = M.copy()
        M_full[0,2] *= (1/downsample_scale)
        M_full[1,2] *= (1/downsample_scale)
    else:
        M_full = M

    h, w = ref_img.shape[:2]
    aligned_img = cv2.warpAffine(img_to_align, M_full, (w, h))
    return aligned_img

def collect_all_images_for_plot(plot_no):
    images = []
    for folder in input_folders:
        for ext in ["JPG", "jpg"]:
            # collect all images matching plot_no_*.JPG or plot_no_*.jpg
            images.extend(glob(os.path.join(folder, f"{plot_no}_*.{ext}")))
    return images

def process_images_auto(image_paths):
    if not image_paths:
        return None

    ref_img = cv2.imread(image_paths[0])
    if ref_img is None:
        return None

    warped_images = [ref_img.astype(np.float32) / 255.0]

    for path in image_paths[1:]:
        img = cv2.imread(path)
        if img is None:
            continue

        aligned = auto_align_images(ref_img, img)
        if aligned is not None:
            warped_images.append(aligned.astype(np.float32) / 255.0)

    if len(warped_images) == 0:
        return None

    avg_img = np.mean(warped_images, axis=0)
    return (avg_img * 255).astype(np.uint8)

def process_plot(plot_no, image_paths):
    print(f"\n--- Processing Plot {plot_no} ---")
    color = plot_colors.get(plot_no, (0, 0, 255))

    rgb_paths = [p for p in image_paths if get_image_type(p) == "RGB"]
    thermal_paths = [p for p in image_paths if get_image_type(p) == "THERMAL"]

    if rgb_paths:
        print(f"  {len(rgb_paths)} RGB images")
        rgb_overlay = process_images_auto(rgb_paths)
        if rgb_overlay is not None:
            cv2.imwrite(os.path.join(output_folder, f"plot_{plot_no}_RGB_overlay.jpg"), rgb_overlay)
        else:
            print("  Failed to align/average RGB images")
    else:
        print("  No RGB images found")

    if thermal_paths:
        print(f"  {len(thermal_paths)} THERMAL images")
        thermal_overlay = process_images_auto(thermal_paths)
        if thermal_overlay is not None:
            cv2.imwrite(os.path.join(output_folder, f"plot_{plot_no}_THERMAL_overlay.jpg"), thermal_overlay)
        else:
            print("  Failed to align/average THERMAL images")
    else:
        print("  No THERMAL images found")

def superimpose_final():
    for image_type in ["RGB", "THERMAL"]:
        paths = sorted(glob(os.path.join(output_folder, f"plot_*_{image_type}_overlay.jpg")))
        if not paths:
            continue
        imgs = [cv2.imread(p).astype(np.float32) / 255.0 for p in paths if cv2.imread(p) is not None]
        if len(imgs) == 0:
            print(f"No {image_type} overlay images to average.")
            continue
        final = np.mean(imgs, axis=0)
        final_img = (final * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_folder, f"FINAL_superimposed_{image_type}.jpg"), final_img)
        print(f"  Saved final superimposed {image_type} overlay for all plots.")

# --- MAIN ---
if __name__ == "__main__":
    print("Starting improved ORB-based temporal alignment...\n")
    for plot_no in plot_numbers:
        all_images = collect_all_images_for_plot(plot_no)
        if all_images:
            process_plot(plot_no, all_images)
        else:
            print(f"No images found for plot {plot_no}")
    print("\nGenerating final overlays...")
    superimpose_final()
    print("Done.")
