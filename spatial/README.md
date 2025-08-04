# Image Alignment for Agricultural Plots

This repository contains scripts and workflows for **spatial** and **temporal alignment** of drone-captured RGB and thermal imagery of agricultural plots.

---

## 📍 Spatial Alignment

Spatial alignment refers to aligning multiple image *modalities* (e.g., RGB and thermal) taken from the **same time** but captured with different sensors or slight positional differences. The goal is to ensure the corresponding pixel regions in both modalities match precisely.

### Tasks:
- Manually select 4 plot corners in both RGB and thermal images.
- Apply homography to warp thermal images to align with RGB.
- Save aligned overlays with corner markers.
