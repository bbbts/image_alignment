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

output_folder = "/home/AD.UNLV.EDU/bhattb3/snow_melt/TEMPORAL_fixed_dst/"
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

def get_image_type_and_size(filename):
    img = cv2.imread(filename)
    if img is None:
        return None, None
    h, w = img.shape[:2]
    if (w, h) == RGB_SIZE:
        return "RGB", (w, h)
    elif (w, h) == THERMAL_SIZE:
        return "THERMAL", (w, h)
    else:
        return ("RGB" if w > 1000 else "THERMAL"), (w, h)

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

def warp_image_and_points(img, src_pts):
    h, w = img.shape[:2]
    dst_rect = np.array([
        [1000, 1000],
        [w - 1000, 1000],
        [w - 1000, h - 1000],
        [1000, h - 1000]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_rect)
    warped_img = cv2.warpPerspective(img, H, (w, h))

    # Transform corner points
    src_pts_homog = np.hstack([src_pts, np.ones((4, 1))])
    warped_pts = (H @ src_pts_homog.T).T
    warped_pts = warped_pts[:, :2] / warped_pts[:, 2:]

    return warped_img, warped_pts

def draw_dots(img, points, color_bgr, radius):
    for (x, y) in points:
        cv2.circle(img, (int(round(x)), int(round(y))), radius=radius, color=color_bgr, thickness=-1)
    return img

def process_image_list(image_paths, plot_color):
    if not image_paths:
        return None

    warped_images = []
    num_images = len(image_paths)
    print(f"   Processing {num_images} images")

    for idx, img_path in enumerate(sorted(image_paths)):
        img = cv2.imread(img_path)
        if img is None:
            print(f"    Skipping unreadable image: {img_path}")
            continue

        print(f"    Selecting corners for: {os.path.basename(img_path)}")
        src_pts = select_4_points(img)
        warped_img, warped_pts = warp_image_and_points(img, src_pts)

        # Style the dot based on time step
        max_radius = 80
        min_radius = 8
        radius = int(max_radius - (max_radius - min_radius) * (idx / max(1, num_images - 1)))

        base_color = np.array(plot_color, dtype=np.float32)
        min_brightness = 0.4
        max_brightness = 1.0
        brightness_factor = min_brightness + (max_brightness - min_brightness) * (idx / max(1, num_images - 1))
        adjusted_color = tuple(np.clip(base_color * brightness_factor, 0, 255).astype(np.uint8).tolist())

        warped_img = draw_dots(warped_img, warped_pts, adjusted_color, radius)

        warped_images.append(warped_img.astype(np.float32) / 255.0)

    if not warped_images:
        return None

    avg_img = np.mean(warped_images, axis=0)
    avg_img = np.clip(avg_img * 255, 0, 255).astype(np.uint8)
    return avg_img

def collect_first_images_by_plot():
    plot_image_map = {plot: [] for plot in plot_numbers}

    for folder in input_folders:
        all_images = sorted(glob(os.path.join(folder, "*.JPG")))
        seen_plots = set()

        for img_path in all_images:
            basename = os.path.basename(img_path)
            try:
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
    rgb_paths = [p for p in image_paths if get_image_type_and_size(p)[0] == "RGB"]
    thermal_paths = [p for p in image_paths if get_image_type_and_size(p)[0] == "THERMAL"]

    if rgb_paths:
        print(f"Processing RGB images for plot {plot_no}...")
        rgb_overlay = process_image_list(rgb_paths, color)
        if rgb_overlay is not None:
            out_path_rgb = os.path.join(output_folder, f"plot_{plot_no}_RGB_overlay.jpg")
            cv2.imwrite(out_path_rgb, rgb_overlay)

    if thermal_paths:
        print(f"Processing Thermal images for plot {plot_no}...")
        thermal_overlay = process_image_list(thermal_paths, color)
        if thermal_overlay is not None:
            out_path_thermal = os.path.join(output_folder, f"plot_{plot_no}_THERMAL_overlay.jpg")
            cv2.imwrite(out_path_thermal, thermal_overlay)

# --- MAIN ---
if __name__ == "__main__":
    print("Starting homography-based image alignment across multiple months...\n")
    plot_images = collect_first_images_by_plot()
    for plot_no, image_paths in plot_images.items():
        process_plot(plot_no, image_paths)

    print("\n? All plots processed. Overlays saved in:", output_folder)
