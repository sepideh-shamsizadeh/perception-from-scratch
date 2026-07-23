import json

import cv2
import numpy as np
import pytest

import detection
import evaluate_tracking as et


def write_blank_image(path, size=(50, 50)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def write_annotation(path, boxes):
    people = [
        {
            "personID": person_id,
            "views": [{
                "viewNum": 0,
                "xmin": box[0],
                "ymin": box[1],
                "xmax": box[2],
                "ymax": box[3],
            }],
        }
        for person_id, box in boxes.items()
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(people))


class FakeTrack:
    def __init__(self, track_id, box, confirmed=True):
        self.track_id = track_id
        self._box = box
        self._confirmed = confirmed

    def is_confirmed(self):
        return self._confirmed

    def to_ltrb(self):
        return self._box


class FakeTracker:
    """Returns one confirmed track per detection, keyed by call order."""

    def __init__(self, track_id="1"):
        self.track_id = track_id

    def update_tracks(self, detections, frame=None):
        tracks = []

        for left, top, width, height in (box for box, _, _ in detections):
            tracks.append(
                FakeTrack(self.track_id, [left, top, left + width, top + height])
            )

        return tracks


class FakeArray:
    """Mimics the .cpu().numpy() chain used on ultralytics tensors."""

    def __init__(self, values):
        self._array = np.array(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class FakeBoxes:
    def __init__(self, box, confidence):
        self.xyxy = FakeArray([box])
        self.conf = FakeArray([confidence])


class FakeResult:
    def __init__(self, box, confidence):
        self.boxes = FakeBoxes(box, confidence)
        self.speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5}


class FakeModel:
    """Always returns the same fixed detection, ignoring the input frame."""

    def __init__(self, box, confidence=0.9):
        self.box = box
        self.confidence = confidence

    def predict(self, **kwargs):
        return [FakeResult(self.box, self.confidence)]


# ---------------------------------------------------------
# camera_to_view_number
# ---------------------------------------------------------


def test_camera_to_view_number_parses_index():
    assert et.camera_to_view_number("C1") == 0
    assert et.camera_to_view_number("C7") == 6


# ---------------------------------------------------------
# build_distance_matrix
# ---------------------------------------------------------


def test_build_distance_matrix_pairs_overlapping_boxes():
    ground_truth_boxes = [[0, 0, 10, 10]]
    track_boxes = [[0, 0, 10, 10], [100, 100, 110, 110]]

    distances = et.build_distance_matrix(
        ground_truth_boxes, track_boxes, iou_threshold=0.5
    )

    assert distances.shape == (1, 2)
    assert distances[0, 0] == pytest.approx(0.0)
    assert np.isnan(distances[0, 1])


def test_build_distance_matrix_respects_iou_threshold():
    # IoU here is 25/175, below a 0.5 threshold.
    ground_truth_boxes = [[0, 0, 10, 10]]
    track_boxes = [[5, 5, 15, 15]]

    distances = et.build_distance_matrix(
        ground_truth_boxes, track_boxes, iou_threshold=0.5
    )

    assert np.isnan(distances[0, 0])


def test_build_distance_matrix_handles_empty_inputs():
    distances = et.build_distance_matrix([], [], iou_threshold=0.5)

    assert distances.shape == (0, 0)


# ---------------------------------------------------------
# evaluate_camera_tracking
# ---------------------------------------------------------


def test_evaluate_camera_tracking_perfect_match_yields_zero_switches(
    tmp_path, monkeypatch
):
    image_dir = tmp_path / "images"
    annotations_dir = tmp_path / "annotations"

    monkeypatch.setattr(detection, "IMAGE_DIR", image_dir)
    monkeypatch.setattr(et, "IMAGE_DIR", image_dir)
    monkeypatch.setattr(et, "ANNOTATIONS_DIR", annotations_dir)

    box = [10, 10, 30, 30]

    for frame_name in ["00000000", "00000005"]:
        write_blank_image(image_dir / "C1" / f"{frame_name}.png")
        write_annotation(annotations_dir / f"{frame_name}.json", {1: box})

    accumulator, evaluated_frames = et.evaluate_camera_tracking(
        model=FakeModel(box=box),
        tracker=FakeTracker(track_id="1"),
        camera_name="C1",
        view_number=0,
    )

    assert evaluated_frames == 2

    events = accumulator.mot_events["Type"].tolist()
    assert events == ["MATCH", "MATCH"]


def test_evaluate_camera_tracking_id_switch_is_recorded(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    annotations_dir = tmp_path / "annotations"

    monkeypatch.setattr(detection, "IMAGE_DIR", image_dir)
    monkeypatch.setattr(et, "IMAGE_DIR", image_dir)
    monkeypatch.setattr(et, "ANNOTATIONS_DIR", annotations_dir)

    box = [10, 10, 30, 30]

    for frame_name in ["00000000", "00000005"]:
        write_blank_image(image_dir / "C1" / f"{frame_name}.png")
        write_annotation(annotations_dir / f"{frame_name}.json", {1: box})

    model = FakeModel(box=box)
    trackers = iter([FakeTracker(track_id="1"), FakeTracker(track_id="2")])

    class SwitchingTracker:
        def update_tracks(self, detections, frame=None):
            return next(trackers).update_tracks(detections, frame=frame)

    accumulator, evaluated_frames = et.evaluate_camera_tracking(
        model=model,
        tracker=SwitchingTracker(),
        camera_name="C1",
        view_number=0,
    )

    assert evaluated_frames == 2
    assert "SWITCH" in accumulator.mot_events["Type"].tolist()


def test_evaluate_camera_tracking_skips_frames_without_annotations(
    tmp_path, monkeypatch
):
    image_dir = tmp_path / "images"
    annotations_dir = tmp_path / "annotations"

    monkeypatch.setattr(detection, "IMAGE_DIR", image_dir)
    monkeypatch.setattr(et, "IMAGE_DIR", image_dir)
    monkeypatch.setattr(et, "ANNOTATIONS_DIR", annotations_dir)

    box = [10, 10, 30, 30]

    write_blank_image(image_dir / "C1" / "00000000.png")
    write_annotation(annotations_dir / "00000000.json", {1: box})
    # Frame with an image but no matching annotation file.
    write_blank_image(image_dir / "C1" / "00000005.png")

    _, evaluated_frames = et.evaluate_camera_tracking(
        model=FakeModel(box=box),
        tracker=FakeTracker(track_id="1"),
        camera_name="C1",
        view_number=0,
    )

    assert evaluated_frames == 1


def test_evaluate_camera_tracking_raises_when_nothing_annotated(
    tmp_path, monkeypatch
):
    image_dir = tmp_path / "images"
    annotations_dir = tmp_path / "annotations"

    monkeypatch.setattr(detection, "IMAGE_DIR", image_dir)
    monkeypatch.setattr(et, "IMAGE_DIR", image_dir)
    monkeypatch.setattr(et, "ANNOTATIONS_DIR", annotations_dir)

    write_blank_image(image_dir / "C1" / "00000000.png")

    with pytest.raises(RuntimeError):
        et.evaluate_camera_tracking(
            model=FakeModel(box=[0, 0, 10, 10]),
            tracker=FakeTracker(),
            camera_name="C1",
            view_number=0,
        )
