import random
import shutil
from pathlib import Path

import numpy as np

from detection_tracking import IMAGE_DIR, get_camera_frame_names


PROJECT_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = PROJECT_DIR / "wildtrack_calibration"

CAMERAS = [f"C{i}" for i in range(1, 8)]
FRAMES_PER_CAMERA = 43
VAL_FRACTION = 0.20
RANDOM_SEED = 42


def select_frame_names(camera_name, frame_count):
    """Select approximately evenly spaced frames from one camera."""
    all_frames = get_camera_frame_names(camera_name)

    if not all_frames:
        raise RuntimeError(
            f"No frames found for camera {camera_name}"
        )

    frame_count = min(frame_count, len(all_frames))

    indices = np.linspace(
        0,
        len(all_frames) - 1,
        frame_count,
        dtype=int,
    )

    # np.linspace can theoretically generate duplicate integer indices.
    unique_indices = sorted(set(indices.tolist()))

    return [all_frames[index] for index in unique_indices]


def copy_samples(samples, destination_dir):
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0

    for camera_name, frame_name in samples:
        source_path = (
            IMAGE_DIR
            / camera_name
            / f"{frame_name}.png"
        )

        destination_path = (
            destination_dir
            / f"{camera_name}_{frame_name}.png"
        )

        if not source_path.exists():
            raise FileNotFoundError(
                f"Calibration source image not found: {source_path}"
            )

        shutil.copy2(source_path, destination_path)
        copied_count += 1

    return copied_count


def write_dataset_yaml():
    yaml_path = CALIBRATION_DIR / "wildtrack.yaml"

    yaml_content = f"""\
path: {CALIBRATION_DIR}

train: images/train
val: images/val

names:
  0: person
"""

    yaml_path.write_text(
        yaml_content,
        encoding="utf-8",
    )

    return yaml_path


def build_calibration_dataset():
    # Remove a previous build so that stale images are not retained.
    if CALIBRATION_DIR.exists():
        shutil.rmtree(CALIBRATION_DIR)

    samples = []

    for camera_name in CAMERAS:
        frame_names = select_frame_names(
            camera_name=camera_name,
            frame_count=FRAMES_PER_CAMERA,
        )

        samples.extend(
            (camera_name, frame_name)
            for frame_name in frame_names
        )

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(samples)

    val_count = round(len(samples) * VAL_FRACTION)

    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    train_dir = CALIBRATION_DIR / "images" / "train"
    val_dir = CALIBRATION_DIR / "images" / "val"

    train_count = copy_samples(
        samples=train_samples,
        destination_dir=train_dir,
    )

    val_count = copy_samples(
        samples=val_samples,
        destination_dir=val_dir,
    )

    yaml_path = write_dataset_yaml()

    print(f"Train: {train_count} images -> {train_dir}")
    print(f"Validation: {val_count} images -> {val_dir}")
    print(f"Total: {train_count + val_count} images")
    print(f"Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    build_calibration_dataset()