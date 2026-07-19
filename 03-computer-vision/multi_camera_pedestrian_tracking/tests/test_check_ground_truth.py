import json

import check_ground_truth as cgt


def write_fixture(tmp_path, monkeypatch, frame_data):
    monkeypatch.setattr(cgt, "annotations_dir", tmp_path)
    (tmp_path / "00000000.json").write_text(json.dumps(frame_data))


def make_view(view_num, xmax=-1, xmin=-1, ymax=-1, ymin=-1):
    return {"viewNum": view_num, "xmax": xmax, "xmin": xmin, "ymax": ymax, "ymin": ymin}


def test_get_box_returns_bbox_for_visible_view(tmp_path, monkeypatch):
    frame_data = [
        {
            "personID": 1,
            "positionID": 0,
            "views": [
                make_view(0, xmax=100, xmin=50, ymax=200, ymin=10),
                make_view(1, xmax=300, xmin=250, ymax=400, ymin=20),
            ],
        }
    ]
    write_fixture(tmp_path, monkeypatch, frame_data)

    assert cgt.get_box("00000000", 0) == [100, 50, 200, 10]


def test_get_box_picks_the_requested_camera(tmp_path, monkeypatch):
    frame_data = [
        {
            "personID": 1,
            "positionID": 0,
            "views": [
                make_view(0, xmax=100, xmin=50, ymax=200, ymin=10),
                make_view(1, xmax=300, xmin=250, ymax=400, ymin=20),
            ],
        }
    ]
    write_fixture(tmp_path, monkeypatch, frame_data)

    assert cgt.get_box("00000000", 1) == [300, 250, 400, 20]


def test_get_box_returns_none_when_not_visible(tmp_path, monkeypatch):
    frame_data = [
        {
            "personID": 1,
            "positionID": 0,
            "views": [make_view(0)],  # sentinel: xmax=-1 means not visible
        }
    ]
    write_fixture(tmp_path, monkeypatch, frame_data)

    assert cgt.get_box("00000000", 0) is None
