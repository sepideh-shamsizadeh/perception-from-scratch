from pathlib import Path
import json


BASE = Path(__file__).parent / "Wildtrack_dataset"
Image_dir = BASE / "image_subsets"
annotations_dir = BASE / "annotations_positions"
calibration_dir = BASE / "calibrations"

def get_box(frame, camera_num):
    with open(annotations_dir/ (frame +".json"), "r", encoding="utf-8") as file:
        data = json.load(file)
    
    cam_fame = data[0]["views"][camera_num]
    bbox = []
    if(cam_fame["xmax"] >0):
        bbox = [cam_fame["xmax"],cam_fame["xmin"], cam_fame["ymax"], cam_fame["ymin"]]
        return bbox

def draw_box(boxes, image):
    #TODO: draw boxs on the given image
    pass

def get_image(camera_num, frame):    
    #TODO: read image for each camera numbera and file name
    pass

def read_annotation_file():
    with open(annotations_dir/"00000000.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    print(len(data))
    print(get_box("00000000", 1))


read_annotation_file()