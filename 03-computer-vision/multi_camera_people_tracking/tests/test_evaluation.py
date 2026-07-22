import pytest

import evaluation as ev


def test_calculate_iou_full_overlap():
    box = [0, 0, 10, 10]

    assert ev.calculate_iou(box, box) == 1.0


def test_calculate_iou_no_overlap():
    assert ev.calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_calculate_iou_partial_overlap():
    # pred area=100, gt area=100, intersection=5x5=25, union=175
    iou = ev.calculate_iou([0, 0, 10, 10], [5, 5, 15, 15])

    assert iou == pytest.approx(25 / 175)


def test_calculate_iou_zero_area_box_returns_zero():
    assert ev.calculate_iou([0, 0, 0, 0], [0, 0, 10, 10]) == 0.0


def test_match_predictions_matches_overlapping_boxes():
    predictions = [{"box": [0, 0, 10, 10], "confidence": 0.9}]
    ground_truth = [{"person_id": 1, "box": [0, 0, 10, 10]}]

    matches, false_positives, false_negatives = ev.match_predictions(
        predictions, ground_truth
    )

    assert len(matches) == 1
    assert matches[0]["iou"] == 1.0
    assert false_positives == []
    assert false_negatives == []


def test_match_predictions_no_overlap_gives_fp_and_fn():
    predictions = [{"box": [0, 0, 10, 10], "confidence": 0.9}]
    ground_truth = [{"person_id": 1, "box": [100, 100, 110, 110]}]

    matches, false_positives, false_negatives = ev.match_predictions(
        predictions, ground_truth
    )

    assert matches == []
    assert len(false_positives) == 1
    assert len(false_negatives) == 1


def test_match_predictions_processes_all_predictions_not_just_first():
    # Regression test: match_predictions used to `return` from inside the
    # for-loop, so only the first prediction was ever checked against
    # ground truth.
    predictions = [
        {"box": [0, 0, 10, 10], "confidence": 0.5},
        {"box": [100, 100, 110, 110], "confidence": 0.9},
    ]
    ground_truth = [
        {"person_id": 1, "box": [0, 0, 10, 10]},
        {"person_id": 2, "box": [100, 100, 110, 110]},
    ]

    matches, false_positives, false_negatives = ev.match_predictions(
        predictions, ground_truth
    )

    assert len(matches) == 2
    assert false_positives == []
    assert false_negatives == []


def test_match_predictions_prefers_higher_confidence_for_same_ground_truth():
    predictions = [
        {"box": [0, 0, 10, 10], "confidence": 0.4},
        {"box": [1, 1, 11, 11], "confidence": 0.95},
    ]
    ground_truth = [{"person_id": 1, "box": [0, 0, 10, 10]}]

    matches, false_positives, false_negatives = ev.match_predictions(
        predictions, ground_truth, iou_threshold=0.5
    )

    assert len(matches) == 1
    assert matches[0]["prediction"]["confidence"] == 0.95
    assert len(false_positives) == 1
    assert false_positives[0]["confidence"] == 0.4
    assert false_negatives == []


def test_match_predictions_respects_iou_threshold():
    # IoU here is 50 / 150 = 0.333, below the 0.5 threshold.
    predictions = [{"box": [0, 0, 10, 10], "confidence": 0.9}]
    ground_truth = [{"person_id": 1, "box": [5, 0, 15, 10]}]

    matches, false_positives, false_negatives = ev.match_predictions(
        predictions, ground_truth, iou_threshold=0.5
    )

    assert matches == []
    assert len(false_positives) == 1
    assert len(false_negatives) == 1


def test_match_predictions_handles_empty_inputs():
    matches, false_positives, false_negatives = ev.match_predictions([], [])

    assert matches == []
    assert false_positives == []
    assert false_negatives == []
