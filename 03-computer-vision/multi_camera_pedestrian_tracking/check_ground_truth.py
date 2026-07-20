from pathlib import Path
import json
import cv2


BASE = Path(__file__).parent / "Wildtrack_dataset"
Image_dir = BASE / "Image_subsets"
annotations_dir = BASE / "annotations_positions"
calibration_dir = BASE / "calibrations"

def get_box(cam_frame):
    bbox = []
    if(cam_frame["xmax"] >0):
        bbox = [cam_frame["xmin"],cam_frame["ymin"], cam_frame["xmax"], cam_frame["ymax"]]
    return bbox


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

def read_annotation_file():
    with open(annotations_dir/"00000000.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    print(len(data))
    image = get_image(1, "00000000.png")

    for i in range(0, len(data)):
        id = data[i]["personID"]
        cam_frame = data[i]["views"][0]
        
        boxes= get_box(cam_frame)
        color = color_for_person(id)
        image = draw_box(boxes, image, color, id)

    cv2.imshow("00000000", image)
    cv2.waitKey(0)


read_annotation_file()