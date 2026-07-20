import cv2
import numpy as np

import check_ground_truth as cgt


def make_view(view_num, xmax=-1, xmin=-1, ymax=-1, ymin=-1):
    return {"viewNum": view_num, "xmax": xmax, "xmin": xmin, "ymax": ymax, "ymin": ymin}


def test_get_box_returns_bbox_for_visible_view():
    view = make_view(0, xmax=100, xmin=50, ymax=200, ymin=10)

    assert cgt.get_box(view) == [50, 10, 100, 200]


def test_get_box_returns_empty_list_when_not_visible():
    view = make_view(0)  # sentinel: xmax=-1 means not visible

    assert cgt.get_box(view) == []


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
