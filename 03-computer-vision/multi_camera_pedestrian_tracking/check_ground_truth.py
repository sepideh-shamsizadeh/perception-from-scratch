from pathlib import Path
import json
import cv2


BASE = Path(__file__).parent / "Wildtrack_dataset"
Image_dir = BASE / "Image_subsets"
annotations_dir = BASE / "annotations_positions"
calibration_dir = BASE / "calibrations"



def color_for_person(person_id):
    blue = (person_id*37)% 256
    green = (person_id*67)% 256
    red = (person_id*97)%256

    return blue, green, red


def draw_box(boxes, image, color, id):
    if(len(boxes)>0):
        image = cv2.rectangle(image, (boxes[0], boxes[1]), (boxes[2],boxes[3]), color, 2)
        image = cv2.putText(image, str(id), (boxes[0]-1, boxes[1]-1), cv2.FONT_HERSHEY_COMPLEX, 1, color, 1)
        
    return image
    

def get_image(camera_num, image_name):    
    camera_path = "C"+ str(camera_num)
    image = cv2.imread(Image_dir/camera_path/image_name)

    return image


def get_box(view):
    box = [
        view["xmin"],
        view["ymin"],
        view["xmax"],
        view["ymax"],
    ]

    if min(box) < 0:
        return None

    if box[2] <= box[0] or box[3] <= box[1]:
        return None

    return box


def read_annotation_file(annotation_path, view_number):
    with open(annotation_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    annotations = []

    for person in data:
        view = person["views"][view_number]
        box = get_box(view)

        if box is None:
            continue

        annotations.append({
            "person_id": person["personID"],
            "box": box,
        })

    return annotations

