import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from evaluation import match_predictions
from check_ground_truth import read_annotation_file


BASE = Path(__file__).parent / "Wildtrack_dataset"
IMAGE_DIR = BASE / "Image_subsets"
ANNOTATIONS_DIR = BASE / "annotations_positions"
OUTPUT_DIR = Path(__file__).parent / "eval_output"


def run_detection(model, image_path):
    start_time = time.perf_counter()

    results = model.predict(
        source=image_path,
        classes=[0],
        conf=0.20,
        imgsz=1920,
        device="cpu",
        save=False,
        verbose=False,
    )

    inference_time = time.perf_counter() - start_time

    result = results[0]

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    predictions = []

    for box, confidence in zip(boxes, confidences):
        predictions.append({
            "box": box.tolist(),
            "confidence": float(confidence),
        })

    return predictions, inference_time


def draw_evaluation(image_path, ground_truth, predictions, output_path):
    image = cv2.imread(str(image_path))

    for gt in ground_truth:
        x1, y1, x2, y2 = (int(value) for value in gt["box"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"GT {gt['person_id']}",
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    for prediction in predictions:
        x1, y1, x2, y2 = (int(value) for value in prediction["box"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            image,
            f"{prediction['confidence']:.2f}",
            (x1, y2 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


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

    predictions, inference_time = run_detection(
        model=model,
        image_path=image_path,
    )

    ground_truth = read_annotation_file(
        annotation_path=annotation_path,
        view_number=view_number,
    )

    if save_visualization:
        draw_evaluation(
            image_path=image_path,
            ground_truth=ground_truth,
            predictions=predictions,
            output_path=OUTPUT_DIR / f"{camera_name}_{frame_name}.png",
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
        if precision_denominator
        else 0.0
    )

    recall = (
        true_positive_count / recall_denominator
        if recall_denominator
        else 0.0
    )

    mean_iou = (
        sum(match["iou"] for match in matches)
        / len(matches)
        if matches
        else 0.0
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
        "inference_time": inference_time,
        "matches": matches,
    }


def evaluate_all_cameras(
    model,
    frame_name,
    camera_count=7,
    iou_threshold=0.5,
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
        )
        all_metrics[camera_name] = metrics

        print(f"--- {camera_name} ---")
        print("Predictions:", metrics["predictions"])
        print("Ground truth:", metrics["ground_truth"])
        print("True positives:", metrics["true_positives"])
        print("False positives:", metrics["false_positives"])
        print("False negatives:", metrics["false_negatives"])
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"Mean IoU: {metrics['mean_iou']:.3f}")
        print(f"Inference time: {metrics['inference_time']:.3f}s")

    return all_metrics


model = YOLO("yolo26n.pt")

evaluate_all_cameras(
    model=model,
    frame_name="00000000",
)