# use the signal detection theory — (dprime, maybe use this) to set the threshold for the simialrity. 
# Use the threhold for that target category, check whether this target is in this image. 
import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch
from gensim.models import KeyedVectors
import sys
import numpy as np
import pandas as pd
import json
from src.YOLO_utils import * 

# Paths
COCO_img_dir = "images/"
glove_path = "glove6B/glove.6B.300d.word2vec.txt"
thresholds_path = "output/category_thresholds.csv"

img_task = "bottle"
img_name = "000000009527.jpg"
img_path = os.path.join(COCO_img_dir, img_task, img_name)
# img_path = os.path.join('houses', 'house1.jpg')

output_dir = "output/"

# Load model, read the thresholds file
os.makedirs(output_dir, exist_ok=True)
YOLO_model = YOLO('yolov8x.pt')
glove_model = KeyedVectors.load_word2vec_format(glove_path)
thresholds_df = pd.read_csv(thresholds_path)

# ============ Prepare ======================
# ======= target word, similarity threshold === vegetable, pumpkin, vehicle
# if want to test many target words, 1. check whether in 18 categories, 2. set relative thred.
target_word = "bowl"
if target_word in thresholds_df["category"].values:
    print(target_word, " is in our 18 categories, has trained threshold.")
    threshold = thresholds_df[thresholds_df["category"] == target_word]["midpoint_threshold"].values[0]
else:
    # no analyzed threshold for this target word 
    threshold = 0
print("Threshold used here is:", threshold)

# ========== Prepare ==========
img = Image.open(img_path).convert("RGB")
names = YOLO_model.names


# ========== RUN YOLO model on ONE image ==============
# verbose=False -> to close the print for YOLO log output infor.                         
YOLOresutls = YOLO_model(img_path, imgsz=1024, augment=True, verbose=False)
boxes = YOLOresutls[0].boxes.xyxy.cpu().numpy()

# ============= save patches/draw boxes on img (for checking, no need later) ==========
# # save patches
# print(f"Number of box detected: {len(boxes)} ")
# save_YOLO_patches(img, boxes)

# # draw boxes
# draw_YOLOboxes_on_Img(img_path, boxes, names, YOLOresutls)

# =========== get unique YOLO detected labels ================
YOLOclass_ids = YOLOresutls[0].boxes.cls.cpu().numpy().astype(int)
YOLO_labels = [names[i] for i in YOLOclass_ids]   # change index -> text labels
YOLO_labels_unique = list(set(YOLO_labels))     # unique labels. 
print("labels unique", YOLO_labels_unique)

# =============== get similarities of yolo labels vs. target word, For ONE image =========
labels_sims = labels_vs_target_similarity_glove(glove_model, YOLO_labels_unique, target_word)

# find the index that sim >= threshold, and store idx in list
overThred_idx = [i for i, val in enumerate(labels_sims) if val >= threshold]
# ======= output result ======
print("\n=== Final Judgment ===")
if overThred_idx!=[]:
    print(f"✅ image contains the target word. As detected labels over the threshold.")
    print("Target word match with: ", [YOLO_labels_unique[i] for i in overThred_idx])
else:
    print("❌ NO target word object in this image.")




    