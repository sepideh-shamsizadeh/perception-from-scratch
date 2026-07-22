import json

import cv2
import numpy as np

import check_ground_truth as cgt


def make_view(view_num, xmax=-1, xmin=-1, ymax=-1, ymin=-1):
    return {"viewNum": view_num, "xmax": xmax, "xmin": xmin, "ymax": ymax, "ymin": ymin}


def make_person(person_id, views):
    return {"personID": person_id, "positionID": 0, "views": views}


def test_get_box_returns_bbox_for_visible_view():
    view = make_view(0, xmax=100, xmin=50, ymax=200, ymin=10)

    assert cgt.get_box(view) == [50, 10, 100, 200]


def test_get_box_returns_none_when_not_visible():
    view = make_view(0)  # sentinel: xmax=-1 means not visible

    assert cgt.get_box(view) is None


def test_get_box_returns_none_for_zero_width_box():
    view = make_view(0, xmin=50, xmax=50, ymin=10, ymax=200)

    assert cgt.get_box(view) is None


def test_get_box_returns_none_for_zero_height_box():
    view = make_view(0, xmin=50, xmax=100, ymin=200, ymax=200)

    assert cgt.get_box(view) is None


def test_color_for_person_is_deterministic():
    assert cgt.color_for_person(122) == cgt.color_for_person(122)


def test_color_for_person_differs_for_different_ids():
    assert cgt.color_for_person(1) != cgt.color_for_person(2)


def test_color_for_person_channels_are_valid_byte_values():
    blue, green, red = cgt.color_for_person(255)

    for channel in (blue, green, red):
        assert 0 <= channel < 256


def test_draw_box_leaves_image_unchanged_when_no_box():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = cgt.draw_box([], image, (0, 0, 255), 1)

    assert np.array_equal(result, image)


def test_draw_box_draws_rectangle_on_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    original = image.copy()
    boxes = [10, 10, 60, 60]
    color = (0, 0, 255)

    result = cgt.draw_box(boxes, image, color, 1)

    assert not np.array_equal(result, original)
    assert tuple(result[10, 30]) == color  # top edge of the rectangle


def test_get_image_reads_file_from_camera_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(cgt, "Image_dir", tmp_path)
    camera_dir = tmp_path / "C1"
    camera_dir.mkdir()
    expected = np.full((20, 30, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(camera_dir / "00000000.png"), expected)

    image = cgt.get_image(1, "00000000.png")

    assert image.shape == expected.shape
    assert np.array_equal(image, expected)


def write_annotation_file(tmp_path, people):
    annotation_path = tmp_path / "00000000.json"
    annotation_path.write_text(json.dumps(people))
    return annotation_path


def test_read_annotation_file_returns_box_for_requested_view(tmp_path):
    people = [
        make_person(1, [
            make_view(0, xmax=100, xmin=50, ymax=200, ymin=10),
            make_view(1, xmax=300, xmin=250, ymax=400, ymin=20),
        ]),
    ]
    annotation_path = write_annotation_file(tmp_path, people)

    annotations = cgt.read_annotation_file(annotation_path, view_number=1)

    assert annotations == [{"person_id": 1, "box": [250, 20, 300, 400]}]


def test_read_annotation_file_skips_people_not_visible_in_view(tmp_path):
    people = [
        make_person(1, [make_view(0, xmax=100, xmin=50, ymax=200, ymin=10)]),
        make_person(2, [make_view(0)]),  # not visible: -1 sentinel
    ]
    annotation_path = write_annotation_file(tmp_path, people)

    annotations = cgt.read_annotation_file(annotation_path, view_number=0)

    assert [a["person_id"] for a in annotations] == [1]


def test_read_annotation_file_returns_empty_list_when_nobody_visible(tmp_path):
    people = [make_person(1, [make_view(0)])]
    annotation_path = write_annotation_file(tmp_path, people)

    annotations = cgt.read_annotation_file(annotation_path, view_number=0)

    assert annotations == []
