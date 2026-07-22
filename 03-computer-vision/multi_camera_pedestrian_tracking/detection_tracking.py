import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from evaluation import match_predictions
from check_ground_truth import read_annotation_file


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

BASE = PROJECT_DIR / "Wildtrack_dataset"
IMAGE_DIR = BASE / "Image_subsets"
ANNOTATIONS_DIR = BASE / "annotations_positions"
OUTPUT_DIR = PROJECT_DIR / "eval_output"


# ---------------------------------------------------------
# Device and inference settings
# ---------------------------------------------------------

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Start with 1920 for TensorRT benchmarking.
# Later you can test 1280 or 1920.
IMAGE_SIZE = 1920
CONFIDENCE_THRESHOLD = 0.20


def synchronize_device():
    """Wait for queued CUDA operations before timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_detection(model, image_path):
    """
    Run person detection on one image.

    Returns:
        predictions: list of predicted boxes and confidence values
        inference_time: complete predict-call latency in seconds
        model_speed: Ultralytics preprocessing, inference and postprocessing times
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    synchronize_device()
    start_time = time.perf_counter()

    results = model.predict(
        source=str(image_path),
        classes=[0],
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        save=False,
        verbose=False,
    )

    synchronize_device()
    inference_time = time.perf_counter() - start_time

    result = results[0]

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    predictions = [
        {
            "box": box.tolist(),
            "confidence": float(confidence),
        }
        for box, confidence in zip(boxes, confidences)
    ]

    return predictions, inference_time, result.speed


def draw_evaluation(
    image_path,
    ground_truth,
    predictions,
    output_path,
):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Ground truth in green
    for gt in ground_truth:
        x1, y1, x2, y2 = (
            int(value) for value in gt["box"]
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            f"GT {gt['person_id']}",
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Predictions in red
    for prediction in predictions:
        x1, y1, x2, y2 = (
            int(value) for value in prediction["box"]
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )

        cv2.putText(
            image,
            f"{prediction['confidence']:.2f}",
            (x1, min(y2 + 15, image.shape[0] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(str(output_path), image)

    if not success:
        raise RuntimeError(
            f"Could not save visualization: {output_path}"
        )


def evaluate_frame(
    model,
    camera_name,
    frame_name,
    view_number,
    iou_threshold=0.5,
    save_visualization=True,
):
    image_path = (
        IMAGE_DIR
        / camera_name
        / f"{frame_name}.png"
    )

    annotation_path = (
        ANNOTATIONS_DIR
        / f"{frame_name}.json"
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Annotation not found: {annotation_path}"
        )

    predictions, inference_time, model_speed = run_detection(
        model=model,
        image_path=image_path,
    )

    ground_truth = read_annotation_file(
        annotation_path=annotation_path,
        view_number=view_number,
    )

    matches, false_positives, false_negatives = (
        match_predictions(
            predictions=predictions,
            ground_truth=ground_truth,
            iou_threshold=iou_threshold,
        )
    )

    true_positive_count = len(matches)
    false_positive_count = len(false_positives)
    false_negative_count = len(false_negatives)

    precision_denominator = (
        true_positive_count + false_positive_count
    )

    recall_denominator = (
        true_positive_count + false_negative_count
    )

    precision = (
        true_positive_count / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        true_positive_count / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    mean_iou = (
        sum(match["iou"] for match in matches)
        / len(matches)
        if matches
        else 0.0
    )

    if save_visualization:
        draw_evaluation(
            image_path=image_path,
            ground_truth=ground_truth,
            predictions=predictions,
            output_path=(
                OUTPUT_DIR
                / f"{camera_name}_{frame_name}.png"
            ),
        )

    return {
        "predictions": len(predictions),
        "ground_truth": len(ground_truth),
        "true_positives": true_positive_count,
        "false_positives": false_positive_count,
        "false_negatives": false_negative_count,
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "end_to_end_time": inference_time,
        "model_speed": model_speed,
        "matches": matches,
    }


def get_camera_frame_names(camera_name):
    camera_dir = IMAGE_DIR / camera_name

    if not camera_dir.exists():
        raise FileNotFoundError(
            f"Camera directory not found: {camera_dir}"
        )

    frame_paths = sorted(
        camera_dir.glob("*.png"),
        key=lambda path: int(path.stem),
    )

    return [path.stem for path in frame_paths]


def process_camera_sequence(
    model,
    camera_name,
    maximum_frames=None,
):
    frame_names = get_camera_frame_names(camera_name)

    if maximum_frames is not None:
        frame_names = frame_names[:maximum_frames]

    previous_predictions = None
    sequence = []

    for frame_name in frame_names:
        image_path = (
            IMAGE_DIR
            / camera_name
            / f"{frame_name}.png"
        )

        predictions, inference_time, model_speed = (
            run_detection(
                model=model,
                image_path=image_path,
            )
        )

        sequence.append({
            "frame_name": frame_name,
            "predictions": predictions,
            "end_to_end_time": inference_time,
            "model_speed": model_speed,
            "previous_predictions": previous_predictions,
        })

        previous_predictions = predictions

    return sequence


def evaluate_all_cameras(
    model,
    frame_name,
    camera_count=7,
    iou_threshold=0.5,
    save_visualizations=True,
):
    all_metrics = {}

    for view_number in range(camera_count):
        camera_name = f"C{view_number + 1}"

        metrics = evaluate_frame(
            model=model,
            camera_name=camera_name,
            frame_name=frame_name,
            view_number=view_number,
            iou_threshold=iou_threshold,
            save_visualization=save_visualizations,
        )

        all_metrics[camera_name] = metrics

        speed = metrics["model_speed"]

        print(f"\n--- {camera_name} ---")
        print("Predictions:", metrics["predictions"])
        print("Ground truth:", metrics["ground_truth"])
        print("True positives:", metrics["true_positives"])
        print("False positives:", metrics["false_positives"])
        print("False negatives:", metrics["false_negatives"])
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"Mean IoU: {metrics['mean_iou']:.3f}")
        print(
            "End-to-end time:"
            f" {metrics['end_to_end_time'] * 1000:.2f} ms"
        )
        print(
            "Model inference:"
            f" {speed['inference']:.2f} ms"
        )

    return all_metrics


def warm_up_model(model, image_path, iterations=10):
    print(f"Warming up model for {iterations} iterations...")

    for _ in range(iterations):
        model.predict(
            source=str(image_path),
            classes=[0],
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            save=False,
            verbose=False,
        )

    synchronize_device()
    print("Warm-up complete.")

def summarize_all_cameras(all_metrics):
    total_tp = sum(
        metrics["true_positives"]
        for metrics in all_metrics.values()
    )
    total_fp = sum(
        metrics["false_positives"]
        for metrics in all_metrics.values()
    )
    total_fn = sum(
        metrics["false_negatives"]
        for metrics in all_metrics.values()
    )

    total_predictions = sum(
        metrics["predictions"]
        for metrics in all_metrics.values()
    )
    total_ground_truth = sum(
        metrics["ground_truth"]
        for metrics in all_metrics.values()
    )

    mean_end_to_end_ms = (
        sum(
            metrics["end_to_end_time"]
            for metrics in all_metrics.values()
        )
        / len(all_metrics)
        * 1000
    )

    mean_model_inference_ms = (
        sum(
            metrics["model_speed"]["inference"]
            for metrics in all_metrics.values()
        )
        / len(all_metrics)
    )

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    print("\n=== Overall summary ===")
    print("Total predictions:", total_predictions)
    print("Total ground truth:", total_ground_truth)
    print("Total TP:", total_tp)
    print("Total FP:", total_fp)
    print("Total FN:", total_fn)
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 score: {f1:.3f}")
    print(
        f"Mean end-to-end time: "
        f"{mean_end_to_end_ms:.2f} ms"
    )
    print(
        f"Mean model inference: "
        f"{mean_model_inference_ms:.2f} ms"
    )

    return {
        "predictions": total_predictions,
        "ground_truth": total_ground_truth,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_end_to_end_ms": mean_end_to_end_ms,
        "mean_model_inference_ms": mean_model_inference_ms,
    }

if __name__ == "__main__":
    print("Project directory:", PROJECT_DIR)
    print("Dataset directory:", BASE)
    print("Device:", DEVICE)
    print("Image size:", IMAGE_SIZE)

    if not BASE.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {BASE}"
        )

    # PyTorch model:
    # model_path = PROJECT_DIR / "yolo11n.pt"

    # Later, for TensorRT, replace the line above with:
    model_path = PROJECT_DIR / "yolo11n_int8.engine"
    if model_path.suffix == ".engine":
        model = YOLO(str(model_path), task="detect")
    else:
        model = YOLO(str(model_path))

    frame_name = "00000000"

    warmup_image = (
        IMAGE_DIR
        / "C1"
        / f"{frame_name}.png"
    )

    warm_up_model(
        model=model,
        image_path=warmup_image,
        iterations=10,
    )


    all_metrics = evaluate_all_cameras(
    model=model,
    frame_name=frame_name,
    camera_count=7,
    iou_threshold=0.5,
    save_visualizations=True,
)

summary = summarize_all_cameras(all_metrics)