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
]
output_folder = "/home/AD.UNLV.EDU/bhattb3/snow_melt/temporal_harris_combined_final1/"
os.makedirs(output_folder, exist_ok=True)

RGB_SIZE = (8000, 6000)
THERMAL_SIZE = (640, 512)
plot_numbers = [i for i in range(1, 23) if i not in [14, 18]]

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

def draw_inlier_matches(ref_img, img_to_align, kp1, kp2, matches, inliers):
    inlier_matches = [m for i, m in enumerate(matches) if inliers[i]]
    return cv2.drawMatches(
        ref_img, kp1, img_to_align, kp2, inlier_matches, None,
        matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

def auto_align_images(ref_img, img_to_align, plot_no, idx):
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_img, None)

    if des1 is None or des2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 10:
        return None

    matches = sorted(matches, key=lambda x: x.distance)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

    H, inliers = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    if H is None or inliers is None:
        return None

    # --- Save debug image ---
    vis = draw_inlier_matches(ref_img, img_to_align, kp1, kp2, matches[:50], inliers.ravel().astype(bool))
    debug_path = os.path.join(output_folder, f"plot_{plot_no}_match_debug_{idx}.jpg")
    cv2.imwrite(debug_path, vis)

    aligned_img = cv2.warpPerspective(img_to_align, H, (ref_img.shape[1], ref_img.shape[0]))
    return aligned_img

def collect_all_images_for_plot(plot_no):
    images = []
    for folder in input_folders:
        for ext in ["JPG", "jpg"]:
            images.extend(glob(os.path.join(folder, f"{plot_no}_*.{ext}")))
    return images

def process_images_auto(image_paths, plot_no):
    if not image_paths:
        return None

    ref_img = cv2.imread(image_paths[0])
    if ref_img is None:
        return None

    warped_images = [ref_img.astype(np.float32) / 255.0]

    for idx, path in enumerate(image_paths[1:], start=1):
        img = cv2.imread(path)
        if img is None:
            continue
        aligned = auto_align_images(ref_img, img, plot_no, idx)
        if aligned is not None:
            warped_images.append(aligned.astype(np.float32) / 255.0)

    if len(warped_images) <= 1:
        return None

    avg_img = np.mean(warped_images, axis=0)
    return (avg_img * 255).astype(np.uint8)

def process_plot(plot_no, image_paths):
    print(f"\n--- Processing Plot {plot_no} ---")
    rgb_paths = [p for p in image_paths if get_image_type(p) == "RGB"]

    if rgb_paths:
        print(f"  {len(rgb_paths)} RGB images")
        rgb_overlay = process_images_auto(rgb_paths, plot_no)
        if rgb_overlay is not None:
            cv2.imwrite(os.path.join(output_folder, f"plot_{plot_no}_RGB_overlay.jpg"), rgb_overlay)
        else:
            print("  Failed to align/average RGB images")
    else:
        print("  No RGB images found")

def superimpose_final():
    paths = sorted(glob(os.path.join(output_folder, f"plot_*_RGB_overlay.jpg")))
    imgs = [cv2.imread(p).astype(np.float32) / 255.0 for p in paths if cv2.imread(p) is not None]
    if len(imgs) == 0:
        return
    final = np.mean(imgs, axis=0)
    final_img = (final * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_folder, f"FINAL_superimposed_RGB.jpg"), final_img)
    print("Saved final RGB overlay")

# --- MAIN ---
if __name__ == "__main__":
    print("Starting enhanced debug alignment with inlier visualization...\n")
    for plot_no in plot_numbers:
        all_images = collect_all_images_for_plot(plot_no)
        if all_images:
            process_plot(plot_no, all_images)
        else:
            print(f"No images found for plot {plot_no}")
    print("\nGenerating final overlay...")
    superimpose_final()
    print("Done.")
