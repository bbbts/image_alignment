# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from glob import glob
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime  # <-- Added for literal month name

# --- CONFIGURATION ---
input_folders = [
    "/home/AD.UNLV.EDU/bhattb3/Plot_labeling_NEW/2024-07-19_Renamed/",
    "/home/AD.UNLV.EDU/bhattb3/Plot_labeling_NEW/2024-08-16_Renamed/",
    "/home/AD.UNLV.EDU/bhattb3/Plot_labeling_NEW/2024-09-19_Renamed/",
    "/home/AD.UNLV.EDU/bhattb3/Plot_labeling_NEW/2025-01-22_Renamed/",
    "/home/AD.UNLV.EDU/bhattb3/Plot_labeling_NEW/2025-03-18_Renamed/",
    "/home/AD.UNLV.EDU/bhattb3/Plot_labeling_NEW/2025-05-06_Renamed/",
    "/home/AD.UNLV.EDU/bhattb3/Plot_labeling_NEW/2025-05-29_Renamed/",
    # "OTHER FOLDER PATHS ..."
]

output_folder = "/home/AD.UNLV.EDU/bhattb3/snow_melt/TEMPORAL_mp4_video/"
video_folder = os.path.join(output_folder, "videos")
os.makedirs(video_folder, exist_ok=True)

RGB_SIZE = (8000, 6000)
THERMAL_SIZE = (640, 512)
FPS = 10
SECONDS_PER_IMAGE = 2
FRAMES_PER_IMAGE = FPS * SECONDS_PER_IMAGE

plot_numbers = [i for i in range(1, 23) if i not in [14, 18]]
plot_colors = {
    1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 255, 0), 4: (0, 255, 255),
    5: (255, 0, 255), 6: (255, 255, 0), 7: (128, 0, 128), 8: (0, 128, 255),
    9: (128, 255, 0), 10: (255, 128, 0), 11: (0, 128, 128), 12: (128, 0, 0),
    13: (0, 0, 128), 15: (64, 64, 64), 16: (192, 0, 192), 17: (0, 192, 192),
    19: (100, 100, 0), 20: (0, 100, 100), 21: (150, 150, 50), 22: (50, 150, 150)
}

# --- HELPERS ---
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
    h, w = img.shape[:2]
    scale = min(1200 / w, 800 / h, 1)
    img_disp = cv2.resize(img, (int(w * scale), int(h * scale))).copy()
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((int(x / scale), int(y / scale)))
            cv2.circle(img_disp, (x, y), 10, (0, 0, 255), -1)
            cv2.imshow(window_name, img_disp)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, click_event)

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
    src_pts_homog = np.hstack([src_pts, np.ones((4, 1))])
    warped_pts = (H @ src_pts_homog.T).T
    warped_pts = warped_pts[:, :2] / warped_pts[:, 2:]
    return warped_img, warped_pts

def draw_dots(img, points, color_bgr, radius):
    for (x, y) in points:
        cv2.circle(img, (int(round(x)), int(round(y))), radius=radius, color=color_bgr, thickness=-1)
    return img

def extract_month_label(path):
    return os.path.basename(os.path.dirname(path))[:7]  # e.g., "2024-07"

def extract_month_name(path):
    try:
        folder = os.path.basename(os.path.dirname(path)).split("_")[0]  # "2024-07-19"
        dt = datetime.strptime(folder, "%Y-%m-%d")
        return dt.strftime("%B %Y")  # e.g., "July 2024"
    except:
        return folder

def overlay_text(cvimg, text, subtext=None):
    pil_img = Image.fromarray(cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font_big = ImageFont.truetype(font_path, 110) if os.path.exists(font_path) else ImageFont.load_default()
    font = ImageFont.truetype(font_path, 72) if os.path.exists(font_path) else ImageFont.load_default()
    font_small = ImageFont.truetype(font_path, 48) if os.path.exists(font_path) else ImageFont.load_default()

    # Draw the literal month name (e.g., "July 2024")
    month_name = extract_month_name(text)
    draw.text((50, 50), month_name, font=font_big, fill=(0, 255, 255))

    # Month code (e.g., "2024-07")
    draw.text((50, 180), subtext if subtext else "", font=font, fill=(255, 255, 0))

    # Image filename
    draw.text((50, 270), os.path.basename(text), font=font_small, fill=(255, 200, 200))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

'''
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
'''

# TO ALIGN ONLY THE FIRST IMAGE OF A CERATIN PLOT FROM A CERTAIN I/P FOLDER:
def collect_first_images_by_plot():
    plot_image_map = {plot: [] for plot in plot_numbers}

    for folder in input_folders:
        all_images = sorted(glob(os.path.join(folder, "*.JPG")))
        seen_plots = set()

        for img_path in all_images:
            try:
                plot_id = int(os.path.basename(img_path).split("_")[0])
                if plot_id in plot_numbers and plot_id not in seen_plots:
                    plot_image_map[plot_id].append(img_path)  # Only one per folder
                    seen_plots.add(plot_id)
            except:
                continue

    # Sort each plot’s images by the folder date (YYYY-MM-DD)
    def extract_date(img_path):
        folder = os.path.basename(os.path.dirname(img_path))
        return folder.split("_")[0]  # e.g., '2024-07-19'

    for plot in plot_image_map:
        plot_image_map[plot].sort(key=extract_date)

    return plot_image_map

def process_plot_video(plot_no, image_paths):
    print(f"\n=== Processing Plot {plot_no} ===")
    color = plot_colors.get(plot_no, (0, 0, 255))
    rgb_paths = [p for p in image_paths if get_image_type_and_size(p)[0] == "RGB"]
    if not rgb_paths:
        print("No RGB images found.")
        return

    ref_img = cv2.imread(rgb_paths[0])
    h, w = ref_img.shape[:2]
    video_path = os.path.join(video_folder, f"plot_{plot_no}_aligned.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))

    for idx, path in enumerate(rgb_paths):
        img = cv2.imread(path)
        print(f"  Aligning: {os.path.basename(path)}")
        src_pts = select_4_points(img)
        warped, warped_pts = warp_image_and_points(img, src_pts)
        radius = 20 - int(idx * (15 / max(1, len(rgb_paths)-1)))
        brightness = 0.5 + 0.5 * (idx / max(1, len(rgb_paths)-1))
        adjusted_color = tuple(int(c * brightness) for c in color)
        warped = draw_dots(warped, warped_pts, adjusted_color, radius)

        label = os.path.basename(path)
        month = extract_month_label(path)
        labeled = overlay_text(warped, path, subtext=month)

        for _ in range(FRAMES_PER_IMAGE):
            out.write(labeled)

    out.release()
    print(f"Saved video: {video_path}")

# --- MAIN ---
if __name__ == "__main__":
    print("Starting per-plot video generation...\n")
    plot_images = collect_first_images_by_plot()
    for plot_no, image_paths in plot_images.items():
        process_plot_video(plot_no, image_paths)
    print(f"\nVideos saved in: {video_folder}")
