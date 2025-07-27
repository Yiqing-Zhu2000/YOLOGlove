# test for head 10 examples of COCO image fixations_TP_train_split1.json.
# since the .json also contains the scanpath_fixation which is we don't need,
# 
from PIL import Image, ImageDraw
import os
import json


def draw_bbox_on_image(img_name, bbox, COCO_image_dir, output_dir, label="target", box_color="red", box_width=4):
    """
    draw the bbox on image and save.

    para:
        img_name (str): eg.  "00000036417.jpg"
        bbox (list): [x_min, y_min, width, height]
        image_dir (str): original image folder
        output_dir (str): output image saved folder path
        label (str): target label
        box_color (str): color of the box
        box_width (int)
    """
    # path
    img_path = os.path.join(COCO_image_dir,label, img_name)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, img_name)

    # open image and draw
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # utilize bbox
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x + w, y + h

    # draw bbox and target label
    draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)
    draw.text((x1, y1), label, fill=box_color)

    # save
    img.save(save_path)
    print(f"Image saved to {save_path}")



# ================= RUN =======================
# read the fixations_TP_train_split1.json file and show first 10 samples:
num_samples = 10
COCO_image_dir = "D:/019_2025summer/Datasets/COCOSearch18-images-TP/images/"
output_dir = "./output/COCO/"


with open("D:/019_2025summer/Datasets/COCOSearch18-fixations-TP/coco_search18_fixations_TP_train_split1.json", "r") as f:
        data = json.load(f)
        
# read and draw images
# results = []
# for item in data[:num_samples]:
#     img_name = item["name"]
#     bbox = item["bbox"]
#     target = item["task"]
#     draw_bbox_on_image(img_name,bbox, COCO_image_dir, output_dir, label=target)

# read and draw ONE named image from .json 
target_imgname = "000000300571.jpg"
# fastly find the first match item
target_item = next((item for item in data if item["name"] == target_imgname), None)
if target_item:
    img_name = target_item["name"]
    bbox = target_item["bbox"]
    target = target_item["task"]
    draw_bbox_on_image(img_name, bbox, COCO_image_dir, output_dir, label=target)
else:
    print(f"Image name {target_imgname} not found.")