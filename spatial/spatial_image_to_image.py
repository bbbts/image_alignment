import cv2
import numpy as np
import os
from glob import glob

# --- CONFIGURATION ---
input_folder = "/home/AD.UNLV.EDU/bhattb3/Plot_Labeling/2025-03-18_Renamed_distance_based_NEW/"
output_folder = "/home/AD.UNLV.EDU/bhattb3/snow_melt/2025-03-18_spatial_image_to_image/"
os.makedirs(output_folder, exist_ok=True)

RGB_SIZE = (8000, 6000)    # width x height
THERMAL_SIZE = (640, 512)
plot_numbers = [i for i in range(1, 23) if i not in [14, 18]]

plot_colors = {
    1: (0, 0, 255),       # red
    2: (255, 0, 0),       # blue
    3: (0, 255, 0),       # green
    4: (0, 255, 255),     # yellow
    5: (255, 0, 255),     # pink
    6: (255, 255, 0),     # cyan
    7: (128, 0, 128),     # purple
    8: (0, 128, 255),     # orange
    9: (128, 255, 0),     # lime
    10: (255, 128, 0),
    11: (0, 128, 128),
    12: (128, 0, 0),
    13: (0, 0, 128),
    15: (64, 64, 64),
    16: (192, 0, 192),
    17: (0, 192, 192),
    19: (100, 100, 0),
    20: (0, 100, 100),
    21: (150, 150, 50),
    22: (50, 150, 150)
}

# --- FUNCTIONS ---

def get_image_type_and_size(filename):
    img = cv2.imread(filename)
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

def draw_dots(img, points, color_bgr, radius):
    for (x, y) in points:
        cv2.circle(img, (int(round(x)), int(round(y))), radius=radius, color=color_bgr, thickness=-1)
    return img

def process_image_list(image_paths, plot_color):
    warped_images = []
    warped_points_list = []
    print(f"  Processing {len(image_paths)} images")

    num_images = len(image_paths)
    dst_pts = None  # <-- Store first image corners as fixed destination

    for idx, img_path in enumerate(sorted(image_paths)):
        img = cv2.imread(img_path)
        if img is None:
            print(f"    Skipping unreadable image: {img_path}")
            continue

        image_type, _ = get_image_type_and_size(img_path)

        print(f"    Selecting points in: {os.path.basename(img_path)}")
        src_pts = select_4_points(img)

        if idx == 0:
            # First image: no warp, keep selected points as destination
            dst_pts = src_pts
            warped_img = img.copy()
            warped_pts = dst_pts
        else:
            # Warp all other images to the fixed dst_pts of first image
            H, _ = cv2.findHomography(src_pts, dst_pts)
            warped_img = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))

            # Warp the source points for visualization
            src_pts_homog = np.hstack([src_pts, np.ones((4, 1))])
            warped_pts = (H @ src_pts_homog.T).T
            warped_pts = warped_pts[:, :2] / warped_pts[:, 2:]

        max_radius = 80
        min_radius = 8
        radius = int(max_radius - (max_radius - min_radius) * (idx / (num_images - 1)))

        base_color = np.array(plot_color, dtype=np.float32)
        min_brightness = 0.4
        max_brightness = 1.0
        brightness_factor = min_brightness + (max_brightness - min_brightness) * (idx / (num_images - 1))
        adjusted_color = tuple(np.clip(base_color * brightness_factor, 0, 255).astype(np.uint8).tolist())

        warped_img = draw_dots(warped_img, warped_pts, adjusted_color, radius=radius)

        warped_images.append(warped_img)
        warped_points_list.append(warped_pts)

    if not warped_images:
        return None

    avg_img = np.zeros_like(warped_images[0], dtype=np.float32)
    for wimg in warped_images:
        avg_img += wimg.astype(np.float32) / 255.0
    avg_img /= len(warped_images)
    avg_img = (avg_img * 255).astype(np.uint8)

    return avg_img

def process_plot(plot_no):
    print(f"\n=== Processing Plot {plot_no} ===")
    images = glob(os.path.join(input_folder, f"{plot_no}_*.JPG"))
    if not images:
        print(f"  No images found for plot {plot_no}")
        return

    color = plot_colors.get(plot_no, (0, 0, 255))  # default red if not defined

    rgb_paths = [p for p in images if get_image_type_and_size(p)[0] == "RGB"]
    thermal_paths = [p for p in images if get_image_type_and_size(p)[0] == "THERMAL"]

    if rgb_paths:
        print(f"Processing RGB images for plot {plot_no}...")
        rgb_overlay = process_image_list(rgb_paths, color)
        if rgb_overlay is not None:
            out_path_rgb = os.path.join(output_folder, f"plot_{plot_no}_RGB_overlay.jpg")
            cv2.imwrite(out_path_rgb, rgb_overlay)
            print(f"   Saved RGB overlay: {out_path_rgb}")

    if thermal_paths:
        print(f"Processing Thermal images for plot {plot_no}...")
        thermal_overlay = process_image_list(thermal_paths, color)
        if thermal_overlay is not None:
            out_path_thermal = os.path.join(output_folder, f"plot_{plot_no}_THERMAL_overlay.jpg")
            cv2.imwrite(out_path_thermal, thermal_overlay)
            print(f"   Saved Thermal overlay: {out_path_thermal}")

def superimpose_final_overlays():
    for image_type in ["RGB", "THERMAL"]:
        pattern = os.path.join(output_folder, f"plot_*_{image_type}_overlay.jpg")
        overlay_paths = sorted(glob(pattern))

        if not overlay_paths:
            print(f"No {image_type} overlay images found for final superimpose.")
            continue

        print(f"\nSuperimposing final overlays for {image_type} from {len(overlay_paths)} images...")

        # Read first image to get shape
        base_img = cv2.imread(overlay_paths[0])
        if base_img is None:
            print(f"Failed to read base image {overlay_paths[0]} for {image_type}.")
            continue
        h, w = base_img.shape[:2]

        accumulated = np.zeros((h, w, 3), dtype=np.float32)

        for path in overlay_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"Skipping unreadable image: {path}")
                continue
            if img.shape[:2] != (h, w):
                print(f"Image size mismatch for {path}, skipping.")
                continue
            accumulated += img.astype(np.float32) / 255.0

        accumulated /= len(overlay_paths)
        final_img = np.clip(accumulated * 255, 0, 255).astype(np.uint8)

        final_path = os.path.join(output_folder, f"FINAL_BOSS_superimposed_{image_type}.jpg")
        cv2.imwrite(final_path, final_img)
        print(f"Saved final boss superimposed overlay: {final_path}")

# --- MAIN ---
if __name__ == "__main__":
    print("Starting plot image warping and overlay generation...\n")
    for plot_no in plot_numbers:
        process_plot(plot_no)

    print("\nAll plots processed. Warped overlays saved in:", output_folder)

    # Now create the final superimposed overlays across all plots
    superimpose_final_overlays()
