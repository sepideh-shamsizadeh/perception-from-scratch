import argparse
import colorsys
from pathlib import Path

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from detection import (
    IMAGE_DIR,
    PROJECT_DIR,
    get_camera_frame_names,
    run_detection_on_frame,
)


def to_deepsort_detections(predictions):
    """
    Convert [x1, y1, x2, y2] predictions into DeepSort's expected
    ([left, top, width, height], confidence, class) tuples.
    """
    detections = []

    for prediction in predictions:
        x1, y1, x2, y2 = prediction["box"]

        detections.append((
            [x1, y1, x2 - x1, y2 - y1],
            prediction["confidence"],
            "person",
        ))

    return detections


def get_track_color(track_id):
    """Deterministic BGR color per track id, stable across frames."""
    hue = (hash(str(track_id)) % 360) / 360.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 0.95)

    return int(blue * 255), int(green * 255), int(red * 255)


def draw_tracks(frame, tracks):
    output = frame.copy()

    for track in tracks:
        if not track.is_confirmed():
            continue

        x1, y1, x2, y2 = (int(value) for value in track.to_ltrb())
        color = get_track_color(track.track_id)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            output,
            f"ID {track.track_id}",
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def load_model(model_path):
    if model_path.suffix == ".engine":
        return YOLO(str(model_path), task="detect")

    return YOLO(str(model_path))


def run_tracking_on_camera(
    model,
    camera_name,
    tracker,
    maximum_frames=None,
    output_dir=None,
    save_video=True,
    stream_fps=10.0,
):
    """
    Run detection + DeepSORT tracking over one camera's Wildtrack frame
    sequence and write an annotated video (or per-frame images) of the
    result.
    """
    frame_names = get_camera_frame_names(camera_name)

    if maximum_frames is not None:
        frame_names = frame_names[:maximum_frames]

    if output_dir is None:
        output_dir = PROJECT_DIR / "tracking_output" / camera_name

    output_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None

    try:
        for frame_name in frame_names:
            image_path = IMAGE_DIR / camera_name / f"{frame_name}.png"
            frame = cv2.imread(str(image_path))

            if frame is None:
                raise FileNotFoundError(
                    f"Could not read frame: {image_path}"
                )

            predictions, _, model_speed = run_detection_on_frame(
                model=model,
                frame=frame,
            )

            tracks = tracker.update_tracks(
                to_deepsort_detections(predictions),
                frame=frame,
            )

            confirmed_tracks = [
                track for track in tracks if track.is_confirmed()
            ]

            print(
                f"camera={camera_name} frame={frame_name} "
                f"detections={len(predictions)} "
                f"active_tracks={len(confirmed_tracks)} "
                f"inference={model_speed['inference']:.1f}ms"
            )

            annotated_frame = draw_tracks(frame, tracks)

            if save_video:
                if video_writer is None:
                    height, width = annotated_frame.shape[:2]

                    video_writer = cv2.VideoWriter(
                        str(output_dir / f"{camera_name}_tracked.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        stream_fps,
                        (width, height),
                    )

                video_writer.write(annotated_frame)
            else:
                cv2.imwrite(
                    str(output_dir / f"{frame_name}.jpg"),
                    annotated_frame,
                )
    finally:
        if video_writer is not None:
            video_writer.release()

    return output_dir


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO detection + DeepSORT tracking on a single "
            "Wildtrack camera view."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_DIR.parent.parent / "yolo26n.pt",
    )

    parser.add_argument(
        "--camera",
        type=str,
        default="C1",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Limit the number of frames processed (default: all).",
    )

    parser.add_argument(
        "--stream-fps",
        type=float,
        default=10.0,
        help="Frame rate used for the output video.",
    )

    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save annotated per-frame JPEGs instead of an mp4.",
    )

    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Frames a track survives without a matching detection.",
    )

    parser.add_argument(
        "--n-init",
        type=int,
        default=3,
        help="Consecutive detections required to confirm a new track.",
    )

    parser.add_argument(
        "--max-cosine-distance",
        type=float,
        default=0.2,
        help="Appearance-embedding gating threshold for re-identification.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    if not arguments.model.exists():
        raise FileNotFoundError(f"Model does not exist: {arguments.model}")

    model = load_model(arguments.model)

    tracker = DeepSort(
        max_age=arguments.max_age,
        n_init=arguments.n_init,
        max_cosine_distance=arguments.max_cosine_distance,
    )

    output_dir = run_tracking_on_camera(
        model=model,
        camera_name=arguments.camera,
        tracker=tracker,
        maximum_frames=arguments.frames,
        save_video=not arguments.save_images,
        stream_fps=arguments.stream_fps,
    )

    print(f"\nTracking output written to: {output_dir}")
