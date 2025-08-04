# 📸 Drone-Based Plot Image Alignment: Spatial & Temporal Workflows

This repository provides tools and scripts for processing drone-captured RGB and thermal images of agricultural plots. It supports two core pipelines:

- **Spatial Alignment** — aligns thermal and RGB imagery taken at the same time to a common spatial reference.
- **Temporal Alignment** — aligns and visualizes images of the same plot captured at different time points to observe progression over time.

These workflows are useful for precision agriculture, remote sensing studies, and crop stress analysis using aligned visual and thermal data.

---

## 🚀 Overview

Drone missions are flown over agricultural plots across several months. Each mission captures:

- RGB images at high resolution (e.g., 8000×6000)
- Thermal images (e.g., 640×512)

Each image contains multiple plots. The alignment tools allow a user to:
1. **Select 4 corners of a plot** in each image.
2. Apply a **homography transformation** to warp all images of the same plot to a fixed rectangle.
3. Visualize spatial consistency across sensors and temporal change across months.

---

## 🧭 Spatial Alignment (Same-Time Multimodal)

Spatial alignment is applied when you want to:
- Align thermal and RGB images taken during the **same drone flight**.
- Generate overlays that compare the same plot across **modalities**.

### Key Features:
- Manual 4-point corner selection for each image.
- Homography-based warping into a consistent rectangular frame.
- Overlay of multiple warped images with **colored dot markers** showing corner confidence.
- Output: Averaged images with corners from all frames visually stacked.

### Output:
```
/TEMPORAL_fixed_dst/
├── plot_1_RGB_overlay.jpg
├── plot_1_THERMAL_overlay.jpg
├── plot_2_RGB_overlay.jpg
...
```

---

## 🕒 Temporal Alignment (Across Months)

Temporal alignment allows visualizing how plots change over time.

### Use Cases:
- Comparing plant growth or stress from July to September.
- Visualizing snow melt, irrigation impact, or foliage changes.

### Key Features:
- Automatically selects the **first image** of each plot from each input folder (month).
- Applies consistent corner-based warping.
- Generates:
  - **Stacked overlays** (same plot from multiple months).
  - **MP4 videos** showing aligned images one after another with:
    - File name
    - Folder date (e.g., `2024-07`)
    - Literal month name (e.g., `July 2024`) in large font.

### Output:
```
/TEMPORAL_mp4_video/
└── videos/
    ├── plot_1_aligned.mp4
    ├── plot_2_aligned.mp4
    ...
```

---

## 🗂 Directory Structure

```plaintext
/input_folders/
    ├── 2024-07-19_Renamed/
    ├── 2024-08-16_Renamed/
    ├── 2024-09-19_Renamed/
    └── ...

/TEMPORAL_fixed_dst/            <-- Averaged RGB/thermal overlays per plot
/TEMPORAL_mp4_video/videos/     <-- Per-plot aligned temporal videos
```

---

## 📌 How It Works (Step-by-Step)

1. **User launches the script**.
2. For each image:
   - User clicks 4 plot corners in the order: **Top-Left → Top-Right → Bottom-Right → Bottom-Left**.
3. Image is warped into a standard rectangular plot using homography.
4. Warped images are either:
   - Averaged (spatial overlay).
   - Combined into MP4 videos (temporal alignment).

---

## ✅ Dependencies

Install the required Python packages:

```bash
pip install opencv-python numpy pillow
```

---

## 🧪 Example Use

```python
# Run spatial alignment script
python spatial_overlay.py

# Run temporal alignment & video generation
python temporal_video_generation.py
```

> Make sure to customize `input_folders` in the scripts to match your image source folders.

---

## 💡 Tips

- Press `Esc` or `Enter` once you've clicked the 4 corners.
- Corner dot size and brightness adjust with time to reflect temporal change.
- Video playback holds each image for 2 seconds (adjustable in script).

---

## 🧠 Notes

- You must manually select plot corners each time unless automated alignment is integrated later.
- Only the **first image per plot per folder** is used in temporal alignment to ensure consistency.
- Supports high-resolution imagery efficiently (up to 8K RGB).

---

## 📬 Contact

For questions, feedback, or collaboration, feel free to open an issue or submit a pull request.

---

🛰️ *Empowering field-scale plot monitoring through clean and consistent imagery.*
