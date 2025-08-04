# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from glob import glob

# --- CONFIGURATION ---
input_folder = "/home/AD.UNLV.EDU/bhattb3/Plot_Labeling/2025-03-18_Renamed_distance_based_NEW/"
output_folder = "/home/AD.UNLV.EDU/bhattb3/snow_melt/2025-03-18_spatial_harris_again/"
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
    h, w = img.shape[:2]
    if (w, h) == RGB_SIZE:
        return "RGB"
    elif (w, h) == THERMAL_SIZE:
        return "THERMAL"
    else:
        return "RGB" if w > 1000 else "THERMAL"

def auto_align_images(ref_img, img_to_align):
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)

    # Detect ORB features
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_img, None)

    if des1 is None or des2 is None:
        return None

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 10:
        return None

    matches = sorted(matches, key=lambda x: x.distance)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

    # Compute Homography
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    aligned_img = cv2.warpPerspective(img_to_align, H, (ref_img.shape[1], ref_img.shape[0]))

    return aligned_img





def process_images_auto(image_paths, plot_color):
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

    # Average images
    avg_img = np.mean(warped_images, axis=0)
    return (avg_img * 255).astype(np.uint8)



def process_plot(plot_no):
    print(f"\n--- Processing Plot {plot_no} ---")
    images = glob(os.path.join(input_folder, f"{plot_no}_*.JPG"))
    if not images:
        print("No images found.")
        return

    color = plot_colors.get(plot_no, (0, 0, 255))

    rgb_paths = [p for p in images if get_image_type(p) == "RGB"]
    thermal_paths = [p for p in images if get_image_type(p) == "THERMAL"]

    if rgb_paths:
        rgb_overlay = process_images_auto(rgb_paths, color)
        if rgb_overlay is not None:
            cv2.imwrite(os.path.join(output_folder, f"plot_{plot_no}_RGB_overlay.jpg"), rgb_overlay)

    if thermal_paths:
        thermal_overlay = process_images_auto(thermal_paths, color)
        if thermal_overlay is not None:
            cv2.imwrite(os.path.join(output_folder, f"plot_{plot_no}_THERMAL_overlay.jpg"), thermal_overlay)

def superimpose_final():
    for image_type in ["RGB", "THERMAL"]:
        paths = sorted(glob(os.path.join(output_folder, f"plot_*_{image_type}_overlay.jpg")))
        if not paths:
            continue

        imgs = [cv2.imread(p).astype(np.float32) / 255.0 for p in paths if cv2.imread(p) is not None]
        final = np.mean(imgs, axis=0)
        final_img = (final * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_folder, f"FINAL_superimposed_{image_type}.jpg"), final_img)

# --- MAIN ---
if __name__ == "__main__":
    print("Starting automatic alignment...")
    for plot_no in plot_numbers:
        process_plot(plot_no)

    print("Generating final overlays...")
    superimpose_final()
    print("Done!")
