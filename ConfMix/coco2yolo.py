import json
from collections import defaultdict
from tqdm import tqdm
import os
import sys

"""hyper parameters"""
json_file_path = sys.argv[1]
output_path = sys.argv[2]

"""load json file"""
name_box_id = defaultdict(list)
id_name = dict()
with open(json_file_path, encoding='utf-8') as f:
    data = json.load(f)

"""generate labels"""
def convert_bbox_coco2yolo(bbox, img_width, img_height):
    x_tl, y_tl, w, h = bbox
    dw = 1.0 / img_width
    dh = 1.0 / img_height
    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0
    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh
    return [x, y, w, h]


images = data['images']
annotations = data['annotations']
for ant in tqdm(annotations):
    id = ant['image_id']
    name = images[id]['file_name'][-8:-4]
    width = images[id]['width']
    height = images[id]['height']
    cat = ant['category_id']
    name_box_id[name].append([convert_bbox_coco2yolo(ant['bbox'], width, height), cat])

"""write to txt"""
for key in tqdm(name_box_id.keys()):
    with open(output_path + '/' + key + '.txt', 'w') as f:    
        box_infos = name_box_id[key]
        for info in box_infos:
            f.write(f"{info[1]-1} {info[0][0]:.6f} {info[0][1]:.6f} {info[0][2]:.6f} {info[0][3]:.6f}\n")