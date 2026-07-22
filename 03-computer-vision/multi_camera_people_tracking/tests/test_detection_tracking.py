import json

import cv2
import numpy as np
import pytest

import detection_tracking as dt


class FakeArray:
    """Mimics the .cpu().numpy() chain used on ultralytics tensors."""

    def __init__(self, values):
        self._array = np.array(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = FakeArray(xyxy)
        self.conf = FakeArray(conf)


class FakeResult:
    def __init__(self, xyxy, conf, speed=None):
        self.boxes = FakeBoxes(xyxy, conf)
        self.speed = speed or {"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5}


class FakeModel:
    """Stands in for a YOLO model so tests don't run real inference."""

    def __init__(self, xyxy=None, conf=None, speed=None):
        self.xyxy = xyxy if xyxy is not None else []
        self.conf = conf if conf is not None else []
        self.speed = speed
        self.call_count = 0

    def predict(self, **kwargs):
        self.call_count += 1
        return [FakeResult(self.xyxy, self.conf, self.speed)]


def write_blank_image(path, size=(50, 50)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


# ---------------------------------------------------------
# run_detection
# ---------------------------------------------------------


def test_run_detection_returns_predictions_and_timing(tmp_path):
    image_path = tmp_path / "frame.png"
    write_blank_image(image_path)
    model = FakeModel(xyxy=[[0, 0, 10, 10]], conf=[0.9])

    predictions, inference_time, speed = dt.run_detection(model, image_path)

    assert predictions == [{"box": [0.0, 0.0, 10.0, 10.0], "confidence": 0.9}]
    assert inference_time >= 0
    assert speed["inference"] == 2.0


def test_run_detection_raises_when_image_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        dt.run_detection(FakeModel(), tmp_path / "missing.png")


# ---------------------------------------------------------
# draw_evaluation
# ---------------------------------------------------------


def test_draw_evaluation_saves_output_file(tmp_path):
    image_path = tmp_path / "frame.png"
    write_blank_image(image_path)
    output_path = tmp_path / "out" / "result.png"

    dt.draw_evaluation(
        image_path=image_path,
        ground_truth=[{"person_id": 1, "box": [0, 0, 10, 10]}],
        predictions=[{"box": [5, 5, 15, 15], "confidence": 0.8}],
        output_path=output_path,
    )

    assert output_path.exists()


def test_draw_evaluation_raises_when_image_unreadable(tmp_path):
    with pytest.raises(ValueError):
        dt.draw_evaluation(
            image_path=tmp_path / "missing.png",
            ground_truth=[],
            predictions=[],
            output_path=tmp_path / "out.png",
        )


# ---------------------------------------------------------
# get_camera_frame_names
# ---------------------------------------------------------


def test_get_camera_frame_names_sorts_numerically(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "IMAGE_DIR", tmp_path)
    camera_dir = tmp_path / "C1"
    for frame_number in [20, 5, 100, 0]:
        write_blank_image(camera_dir / f"{frame_number:08d}.png")

    frame_names = dt.get_camera_frame_names("C1")

    assert frame_names == ["00000000", "00000005", "00000020", "00000100"]


def test_get_camera_frame_names_raises_when_camera_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "IMAGE_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        dt.get_camera_frame_names("C1")


# ---------------------------------------------------------
# evaluate_frame
# ---------------------------------------------------------


def setup_frame_fixture(tmp_path, monkeypatch, person_box):
    image_dir = tmp_path / "images"
    annotations_dir = tmp_path / "annotations"
    output_dir = tmp_path / "output"

    monkeypatch.setattr(dt, "IMAGE_DIR", image_dir)
    monkeypatch.setattr(dt, "ANNOTATIONS_DIR", annotations_dir)
    monkeypatch.setattr(dt, "OUTPUT_DIR", output_dir)

    write_blank_image(image_dir / "C1" / "00000000.png")

    annotations_dir.mkdir(parents=True, exist_ok=True)
    people = [{
        "personID": 1,
        "positionID": 0,
        "views": [{
            "viewNum": 0,
            "xmin": person_box[0],
            "ymin": person_box[1],
            "xmax": person_box[2],
            "ymax": person_box[3],
        }],
    }]
    (annotations_dir / "00000000.json").write_text(json.dumps(people))

    return output_dir


def test_evaluate_frame_computes_metrics(tmp_path, monkeypatch):
    output_dir = setup_frame_fixture(tmp_path, monkeypatch, [0, 0, 10, 10])
    model = FakeModel(xyxy=[[0, 0, 10, 10]], conf=[0.9])

    metrics = dt.evaluate_frame(
        model=model,
        camera_name="C1",
        frame_name="00000000",
        view_number=0,
    )

    assert metrics["predictions"] == 1
    assert metrics["ground_truth"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert (output_dir / "C1_00000000.png").exists()


def test_evaluate_frame_skips_visualization_when_disabled(tmp_path, monkeypatch):
    output_dir = setup_frame_fixture(tmp_path, monkeypatch, [0, 0, 10, 10])
    model = FakeModel(xyxy=[[0, 0, 10, 10]], conf=[0.9])

    dt.evaluate_frame(
        model=model,
        camera_name="C1",
        frame_name="00000000",
        view_number=0,
        save_visualization=False,
    )

    assert not (output_dir / "C1_00000000.png").exists()


def test_evaluate_frame_raises_when_image_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "IMAGE_DIR", tmp_path / "images")
    monkeypatch.setattr(dt, "ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr(dt, "OUTPUT_DIR", tmp_path / "output")
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "00000000.json").write_text("[]")

    with pytest.raises(FileNotFoundError):
        dt.evaluate_frame(
            model=FakeModel(),
            camera_name="C1",
            frame_name="00000000",
            view_number=0,
        )


# ---------------------------------------------------------
# process_camera_sequence
# ---------------------------------------------------------


def test_process_camera_sequence_chains_previous_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "IMAGE_DIR", tmp_path)
    camera_dir = tmp_path / "C1"
    write_blank_image(camera_dir / "00000000.png")
    write_blank_image(camera_dir / "00000005.png")
    model = FakeModel(xyxy=[[0, 0, 10, 10]], conf=[0.9])

    sequence = dt.process_camera_sequence(model=model, camera_name="C1")

    assert [item["frame_name"] for item in sequence] == ["00000000", "00000005"]
    assert sequence[0]["previous_predictions"] is None
    assert sequence[1]["previous_predictions"] == sequence[0]["predictions"]


def test_process_camera_sequence_respects_maximum_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "IMAGE_DIR", tmp_path)
    camera_dir = tmp_path / "C1"
    for frame_number in [0, 5, 10]:
        write_blank_image(camera_dir / f"{frame_number:08d}.png")
    model = FakeModel()

    sequence = dt.process_camera_sequence(
        model=model, camera_name="C1", maximum_frames=2
    )

    assert len(sequence) == 2


# ---------------------------------------------------------
# summarize_all_cameras
# ---------------------------------------------------------


def test_summarize_all_cameras_aggregates_metrics():
    all_metrics = {
        "C1": {
            "predictions": 2,
            "ground_truth": 2,
            "true_positives": 2,
            "false_positives": 0,
            "false_negatives": 0,
            "end_to_end_time": 0.1,
            "model_speed": {"inference": 10.0},
        },
        "C2": {
            "predictions": 1,
            "ground_truth": 2,
            "true_positives": 0,
            "false_positives": 1,
            "false_negatives": 2,
            "end_to_end_time": 0.3,
            "model_speed": {"inference": 20.0},
        },
    }

    summary = dt.summarize_all_cameras(all_metrics)

    assert summary["true_positives"] == 2
    assert summary["false_positives"] == 1
    assert summary["false_negatives"] == 2
    assert summary["precision"] == pytest.approx(2 / 3)
    assert summary["recall"] == pytest.approx(2 / 4)
    assert summary["mean_end_to_end_ms"] == pytest.approx((0.1 + 0.3) / 2 * 1000)
    assert summary["mean_model_inference_ms"] == pytest.approx((10.0 + 20.0) / 2)


def test_summarize_all_cameras_handles_zero_predictions():
    all_metrics = {
        "C1": {
            "predictions": 0,
            "ground_truth": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "end_to_end_time": 0.0,
            "model_speed": {"inference": 0.0},
        },
    }

    summary = dt.summarize_all_cameras(all_metrics)

    assert summary["precision"] == 0.0
    assert summary["recall"] == 0.0
    assert summary["f1"] == 0.0


# ---------------------------------------------------------
# warm_up_model
# ---------------------------------------------------------


def test_warm_up_model_calls_predict_iterations_times(tmp_path):
    image_path = tmp_path / "frame.png"
    write_blank_image(image_path)
    model = FakeModel()

    dt.warm_up_model(model=model, image_path=image_path, iterations=3)

    assert model.call_count == 3
