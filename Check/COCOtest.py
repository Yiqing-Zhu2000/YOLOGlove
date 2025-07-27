import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch
from gensim.models import KeyedVectors
import sys
import numpy as np
import json


num_samples = 100
with open("D:/019_2025summer/Datasets/COCOSearch18-fixations-TP/coco_search18_fixations_TP_train_split1.json", "r") as f:
        data = json.load(f)
        
signal_sims = []  # for signal distribution. 
noisy_sims = []   # for noise distribution. 
target_used = "bottle"
for item in data[:num_samples]:
    img_name = item["name"]
    bbox = item["bbox"]
    target = item["task"]  
    if target_used ==  target:  # test got: 100 samples' task == "bottle"
          continue 
    else:
          print("NOOO, this item's target is:", target)
print("DONE!")

# check whether num_samples are all different images. 
names = [item["name"] for item in data[:num_samples]]
unique_names = set(names)

print(f"Total names: {len(names)}")
print(f"Unique names: {len(unique_names)}")
