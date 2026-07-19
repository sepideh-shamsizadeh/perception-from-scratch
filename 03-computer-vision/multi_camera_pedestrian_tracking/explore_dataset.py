"""Explore the Wildtrack multi-camera pedestrian tracking dataset and print
answers to the questions listed in Readme.md.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from PIL import Image

BASE = Path(__file__).parent / "Wildtrack_dataset"
IMAGE_SUBSETS = BASE / "Image_subsets"
ANNOTATIONS = BASE / "annotations_positions"
CALIBRATIONS = BASE / "calibrations"


def get_camera_dirs():
    return sorted(d for d in IMAGE_SUBSETS.iterdir() if d.is_dir())

def number_of_cameras():
    return len(get_camera_dirs())


def image_resolution():
    cam_dirs = get_camera_dirs()
    sample_image = next(cam_dirs[0].glob("*.png"))
    with Image.open(sample_image) as img:
        plt.imshow(img)
        plt.show()
        return img.size  # (width, height)


def images_per_camera():
    return {cam.name: len(list(cam.glob("*.png"))) for cam in get_camera_dirs()}


def image_filename_format():
    cam_dirs = get_camera_dirs()
    sample_names = sorted(p.stem for p in cam_dirs[0].glob("*.png"))[:5]
    return sample_names


def number_of_annotation_files():
    return len(list(ANNOTATIONS.glob("*.json")))


def calibration_file_names():
    return {
        sub.name: sorted(p.name for p in sub.glob("*"))
        for sub in CALIBRATIONS.iterdir()
        if sub.is_dir()
    }


def same_frame_numbers_across_cameras():
    cam_dirs = get_camera_dirs()
    frame_sets = {cam.name: {p.stem for p in cam.glob("*.png")} for cam in cam_dirs}
    reference = next(iter(frame_sets.values()))
    all_same = all(frames == reference for frames in frame_sets.values())
    return all_same, frame_sets


def annotation_file_contents_summary():
    sample_file = sorted(ANNOTATIONS.glob("*.json"))[0]
    with open(sample_file) as f:
        data = json.load(f)

    summary = {
        "file": sample_file.name,
        "type": type(data).__name__,
        "num_entries (people annotated)": len(data),
        "entry_keys": list(data[0].keys()),
        "views_per_entry": len(data[0]["views"]),
        "view_keys": list(data[0]["views"][0].keys()),
        "sample_entry": data[0],
    }
    return summary


def main():
    print("=" * 60)
    print("Number of cameras:", number_of_cameras())

    print("=" * 60)
    w, h = image_resolution()
    print(f"Image resolution: {w}x{h}")

    print("=" * 60)
    print("Number of images per camera:")
    for cam, count in images_per_camera().items():
        print(f"  {cam}: {count}")

    print("=" * 60)
    print("Image filename format (sample from C1):")
    for name in image_filename_format():
        print(f"  {name}")

    print("=" * 60)
    print("Number of annotation files:", number_of_annotation_files())

    print("=" * 60)
    print("Calibration file names:")
    for subdir, files in calibration_file_names().items():
        print(f"  {subdir}/")
        for f in files:
            print(f"    {f}")

    print("=" * 60)
    same, frame_sets = same_frame_numbers_across_cameras()
    print("Every camera has the same frame numbers:", same)
    if not same:
        for cam, frames in frame_sets.items():
            print(f"  {cam}: {len(frames)} frames")

    print("=" * 60)
    print("What one annotation JSON file contains:")
    summary = annotation_file_contents_summary()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
