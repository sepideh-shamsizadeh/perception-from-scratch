import argparse
from pathlib import Path

import cv2
import motmetrics as mm
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from check_ground_truth import read_annotation_file
from deepsort_tracking import load_model, to_deepsort_detections
from detection import (
    ANNOTATIONS_DIR,
    IMAGE_DIR,
    PROJECT_DIR,
    get_camera_frame_names,
    run_detection_on_frame,
)
from evaluation import calculate_iou


def camera_to_view_number(camera_name):
    return int(camera_name.lstrip("C")) - 1


def build_distance_matrix(ground_truth_boxes, track_boxes, iou_threshold):
    """
    motmetrics.distances.iou_matrix calls the numpy-2.0-removed
    np.asfarray, so the IoU distance matrix is built by hand here
    using the same calculate_iou already relied on for detection
    evaluation. Rows are ground truth, columns are tracks; a cell is
    NaN (do-not-pair) when boxes don't overlap enough to match.
    """
    distances = np.full(
        (len(ground_truth_boxes), len(track_boxes)),
        np.nan,
    )

    for row, ground_truth_box in enumerate(ground_truth_boxes):
        for column, track_box in enumerate(track_boxes):
            iou = calculate_iou(track_box, ground_truth_box)

            if iou >= iou_threshold:
                distances[row, column] = 1.0 - iou

    return distances


def evaluate_camera_tracking(
    model,
    tracker,
    camera_name,
    view_number,
    maximum_frames=None,
    iou_threshold=0.5,
):
    """
    Run detection + DeepSORT over one camera's frame sequence and
    accumulate CLEAR MOT statistics against ground-truth person IDs.

    Ground-truth person IDs are stable across the whole Wildtrack
    sequence, so this directly measures how well the tracker keeps a
    single ID on each person over time (as opposed to per-frame
    detection quality, which evaluation.py already covers).
    """
    frame_names = get_camera_frame_names(camera_name)

    if maximum_frames is not None:
        frame_names = frame_names[:maximum_frames]

    accumulator = mm.MOTAccumulator(auto_id=True)
    evaluated_frames = 0

    for frame_name in frame_names:
        annotation_path = ANNOTATIONS_DIR / f"{frame_name}.json"

        if not annotation_path.exists():
            continue

        image_path = IMAGE_DIR / camera_name / f"{frame_name}.png"
        frame = cv2.imread(str(image_path))

        if frame is None:
            raise FileNotFoundError(f"Could not read frame: {image_path}")

        ground_truth = read_annotation_file(
            annotation_path=annotation_path,
            view_number=view_number,
        )

        predictions, _, _ = run_detection_on_frame(
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

        gt_ids = [gt["person_id"] for gt in ground_truth]
        gt_boxes = [gt["box"] for gt in ground_truth]

        track_ids = [track.track_id for track in confirmed_tracks]
        track_boxes = [
            list(track.to_ltrb()) for track in confirmed_tracks
        ]

        distances = build_distance_matrix(
            gt_boxes,
            track_boxes,
            iou_threshold=iou_threshold,
        )

        accumulator.update(gt_ids, track_ids, distances)
        evaluated_frames += 1

    if evaluated_frames == 0:
        raise RuntimeError(
            f"No annotated frames found for {camera_name}."
        )

    return accumulator, evaluated_frames


def summarize_tracking(accumulator, camera_name):
    metrics_host = mm.metrics.create()

    summary = metrics_host.compute(
        accumulator,
        metrics=mm.metrics.motchallenge_metrics,
        name=camera_name,
    )

    return mm.io.render_summary(
        summary,
        formatters=metrics_host.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DeepSORT tracking on a single Wildtrack camera "
            "view against ground-truth person IDs using CLEAR MOT "
            "metrics (MOTA, MOTP, ID switches, ...)."
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
        help="Limit the number of frames evaluated (default: all).",
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Minimum IoU for a track-to-ground-truth match.",
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

    accumulator, evaluated_frames = evaluate_camera_tracking(
        model=model,
        tracker=tracker,
        camera_name=arguments.camera,
        view_number=camera_to_view_number(arguments.camera),
        maximum_frames=arguments.frames,
        iou_threshold=arguments.iou_threshold,
    )

    print(f"Evaluated {evaluated_frames} frames on {arguments.camera}.\n")
    print(summarize_tracking(accumulator, arguments.camera))
