# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from glob import glob

# --- CONFIGURATION ---
input_folders = [
    "/home/AD.UNLV.EDU/bhattb3/Plot_labeling_NEW/2024-07-19_Renamed/",
    "/home/AD.UNLV.EDU/bhattb3/Plot_labeling_NEW/2024-08-16_Renamed/",
    "/home/AD.UNLV.EDU/bhattb3/Plot_labeling_NEW/2024-09-19_Renamed/",
    #"/home/AD.UNLV.EDU/bhattb3/Plot_labeling_NEW/2025-01-22_Renamed/"
]

output_folder = "/home/AD.UNLV.EDU/bhattb3/snow_melt/TEMPORAL_IMAGE_TO_IMAGE"
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
    if img is None:
        return None
    h, w = img.shape[:2]
    if (w, h) == RGB_SIZE:
        return "RGB"
    elif (w, h) == THERMAL_SIZE:
        return "THERMAL"
    else:
        return "RGB" if w > 1000 else "THERMAL"

def select_4_points(img):
    window_name = 'Select 4 corners (ESC/Enter to finish)'
    display_max = 1200
    h, w = img.shape[:2]
    scale = min(display_max / w, display_max / h, 1)
    img_disp = cv2.resize(img, (int(w * scale), int(h * scale))).copy()
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((int(x / scale), int(y / scale)))
            cv2.circle(img_disp, (x, y), 10, (0, 0, 255), -1)
            cv2.imshow(window_name, img_disp)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, click_event)

    print("Click 4 corners of the plot in order: top-left, top-right, bottom-right, bottom-left.")
    print("Then press Enter or ESC to finish.")
    while True:
        cv2.imshow(window_name, img_disp)
        key = cv2.waitKey(20) & 0xFF
        if key in [13, 27] and len(points) == 4:
            break
    cv2.destroyWindow(window_name)
    return np.array(points, dtype=np.float32)

def draw_dots(img, points, color_bgr, radius):
    for (x, y) in points:
        cv2.circle(img, (int(round(x)), int(round(y))), radius=radius, color=color_bgr, thickness=-1)
    return img

def warp_to_reference(ref_img, ref_pts, target_img):
    H, _ = cv2.findHomography(ref_pts, ref_pts)  # dummy if same
    if ref_img.shape != target_img.shape:
        H, _ = cv2.findHomography(select_4_points(target_img), ref_pts)
    return cv2.warpPerspective(target_img, H, (ref_img.shape[1], ref_img.shape[0])), H

def warp_using_reference(ref_img, ref_pts, target_img, target_pts):
    H, _ = cv2.findHomography(target_pts, ref_pts)
    warped_img = cv2.warpPerspective(target_img, H, (ref_img.shape[1], ref_img.shape[0]))

    target_pts_homog = np.hstack([target_pts, np.ones((4, 1))])
    warped_pts = (H @ target_pts_homog.T).T
    warped_pts = warped_pts[:, :2] / warped_pts[:, 2:]

    return warped_img, warped_pts

def process_image_list(image_paths, plot_color):
    if not image_paths:
        return None

    ref_img = cv2.imread(image_paths[0])
    if ref_img is None:
        return None

    print(f" ? Select corners for reference image: {os.path.basename(image_paths[0])}")
    ref_pts = select_4_points(ref_img)

    warped_images = []
    num_images = len(image_paths)

    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"    Skipping unreadable: {path}")
            continue

        if idx == 0:
            warped_img = ref_img.copy()
            warped_pts = ref_pts.copy()
        else:
            print(f" ? Selecting corners for alignment: {os.path.basename(path)}")
            src_pts = select_4_points(img)
            warped_img, warped_pts = warp_using_reference(ref_img, ref_pts, img, src_pts)

        # Visual overlay: shrinking/fading circles
        max_radius = 80
        min_radius = 8
        radius = int(max_radius - (max_radius - min_radius) * (idx / max(1, num_images - 1)))
        brightness = 0.4 + 0.6 * (idx / max(1, num_images - 1))
        dot_color = tuple(np.clip(np.array(plot_color) * brightness, 0, 255).astype(np.uint8).tolist())
        warped_img = draw_dots(warped_img, warped_pts, dot_color, radius)

        warped_images.append(warped_img.astype(np.float32) / 255.0)

    if not warped_images:
        return None

    avg_img = np.mean(warped_images, axis=0)
    return (np.clip(avg_img * 255, 0, 255)).astype(np.uint8)

def collect_first_images_by_plot():
    plot_image_map = {plot: [] for plot in plot_numbers}
    for folder in input_folders:
        all_images = sorted(glob(os.path.join(folder, "*.JPG")))
        seen_plots = set()

        for img_path in all_images:
            try:
                basename = os.path.basename(img_path)
                plot_id = int(basename.split("_")[0])
                if plot_id in plot_numbers and plot_id not in seen_plots:
                    plot_image_map[plot_id].append(img_path)
                    seen_plots.add(plot_id)
            except:
                continue
    return plot_image_map

def process_plot(plot_no, image_paths):
    print(f"\n=== Processing Plot {plot_no} ===")
    color = plot_colors.get(plot_no, (0, 0, 255))
    rgb_paths = [p for p in image_paths if get_image_type(p) == "RGB"]
    thermal_paths = [p for p in image_paths if get_image_type(p) == "THERMAL"]

    if rgb_paths:
        print(f"Processing RGB images...")
        rgb_overlay = process_image_list(rgb_paths, color)
        if rgb_overlay is not None:
            cv2.imwrite(os.path.join(output_folder, f"plot_{plot_no}_RGB_overlay.jpg"), rgb_overlay)

    if thermal_paths:
        print(f"Processing Thermal images...")
        thermal_overlay = process_image_list(thermal_paths, color)
        if thermal_overlay is not None:
            cv2.imwrite(os.path.join(output_folder, f"plot_{plot_no}_THERMAL_overlay.jpg"), thermal_overlay)

# --- MAIN ---
if __name__ == "__main__":
    print("?? Starting temporal image-to-image alignment...")
    plot_images = collect_first_images_by_plot()
    for plot_no, paths in plot_images.items():
        process_plot(plot_no, paths)
    print("\n? Done! Overlays saved in:", output_folder)
